#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio realtime speech-to-text service.

Streams audio to Fish Audio's realtime ASR WebSocket
(``wss://api.fish.audio/v1/asr/realtime``), which transcribes it turn by turn
and decides when the user has finished speaking.
"""

import asyncio
import re
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SpeechControlParamsFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
    UserBackchannelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import FISH_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

try:
    import ormsgpack
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Fish Audio, you need to `uv add "pipecat-ai[fish]"`.')
    raise ImportError(f"Missing module: {e}") from e


FishAudioSTTTurnDetectionMode = Literal["auto", "vad", "manual"]
FishAudioSTTTextFormat = Literal["spoken", "written"]

# Sample rates Fish Audio accepts. Audio at 16 kHz is used as is; every other
# rate is converted on the server.
_SAMPLE_RATES = (8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
_DEFAULT_SEND_SAMPLE_RATE = 16000

# Fish Audio recommends 20-40 ms of audio per message for live audio.
_MIN_CHUNK_SECS = 0.02

# The server closes a session after 30 seconds without a client message.
_KEEPALIVE_SECS = 10.0

# Audio kept while no connection is open, sent once the next one opens.
_MAX_BUFFERED_SECS = 5.0

# Limits of the session's conversation context. A config over them is refused,
# which ends the session.
_MAX_TERMS = 100
_MAX_TERM_CHARS = 32
_MAX_HISTORY_ITEMS = 32
_MAX_HISTORY_CHARS = 2000

_SILENCE_TIMEOUT_RANGE_MS = {
    "auto": (3000, 60000),
    "vad": (300, 60000),
    "manual": (300, 60000),
}
# The server's silence_timeout_ms for each mode, sent explicitly to restore it.
_DEFAULT_SILENCE_TIMEOUT_MS = {"auto": 5000, "vad": 700, "manual": 60000}

# Errors that a retry cannot clear: the session's settings or audio were refused.
_INVALID_SETTINGS_ERROR_TYPES = ("invalid_config", "unsupported_audio")

# Speaker tags in diarized text, such as "<|speaker:0|>".
_SPEAKER_TAG = re.compile(r"<\|speaker:([^|]*)\|>")
# Sound tags, such as "[laughter]", in square, full-width or lenticular brackets.
_SOUND_TAG = re.compile(r"\[[^\[\]]*\]|\uff3b[^\uff3b\uff3d]*\uff3d|\u3010[^\u3010\u3011]*\u3011")


def _sanitize_terms(terms: list[str] | None) -> list[str]:
    """Keep the terms Fish Audio accepts: up to 100 distinct terms of up to 32 characters."""
    if not terms:
        return []
    sanitized: list[str] = []
    seen: set[str] = set()
    too_long = 0
    for raw_term in terms:
        term = raw_term.strip()
        if not term or term in seen:
            continue
        if len(term) > _MAX_TERM_CHARS:
            too_long += 1
            continue
        seen.add(term)
        sanitized.append(term)
    if too_long:
        logger.warning(
            f"Fish Audio STT: dropped {too_long} keyterm(s) over {_MAX_TERM_CHARS} chars"
        )
    if len(sanitized) > _MAX_TERMS:
        logger.warning(
            f"Fish Audio STT: keeping the first {_MAX_TERMS} of {len(sanitized)} keyterms"
        )
        sanitized = sanitized[:_MAX_TERMS]
    return sanitized


def _validate_turn_detection(mode: Any, silence_timeout_ms: int | None):
    """Raise unless the mode is known and ``silence_timeout_ms`` is in its range."""
    allowed = _SILENCE_TIMEOUT_RANGE_MS.get(mode)
    if allowed is None:
        raise ValueError(f"turn_detection_mode must be one of {list(_SILENCE_TIMEOUT_RANGE_MS)}")
    if silence_timeout_ms is None:
        return
    low = allowed[0]
    high = allowed[1]
    if silence_timeout_ms < low or silence_timeout_ms > high:
        raise ValueError(
            f"silence_timeout_ms must be between {low} and {high} in {mode!r} turn detection mode"
        )


def _split_speakers(text: str) -> list[tuple[str | None, str]]:
    """Split diarized text into ``(speaker label, text)`` runs.

    Text before the first speaker tag, or text with no tags, has no label.
    """
    runs: list[tuple[str | None, str]] = []
    label: str | None = None
    position = 0
    for match in _SPEAKER_TAG.finditer(text):
        before = text[position : match.start()].strip()
        if before:
            runs.append((label, before))
        label = match.group(1)
        position = match.end()
    rest = text[position:].strip()
    if rest:
        runs.append((label, rest))
    return runs


def _strip_speaker_tags(text: str) -> str:
    """Remove speaker tags and collapse the whitespace they leave."""
    without_tags = _SPEAKER_TAG.sub(" ", text)
    return " ".join(without_tags.split())


def _words_only(text: str) -> str:
    """The words of a transcript, without speaker or sound tags."""
    without_speakers = _SPEAKER_TAG.sub(" ", text)
    without_sounds = _SOUND_TAG.sub(" ", without_speakers)
    return " ".join(without_sounds.split())


class FishAudioSTTRequestError(Exception):
    """Fish Audio refused the session's settings or audio format.

    Carries an HTTP-style status code, so the error is classified like a
    rejected request.
    """

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize the error.

        Args:
            message: The error message.
            status_code: The HTTP-style status the server reported.
        """
        super().__init__(message)
        self.status_code = status_code


class _TextFormatRefusedError(ConnectionError):
    """The session's model refused ``text_format``; a new session goes without it."""


@dataclass
class FishAudioSTTSettings(STTSettings):
    """Settings for FishAudioSTTService.

    Fish Audio detects the language of each turn, so ``language`` is not used.

    Parameters:
        keyterms: Names, products and jargon to recognize. Up to 100 terms of
            up to 32 characters; longer terms are dropped. Updated on the open
            session.
        turn_detection_mode: How Fish Audio ends a turn. ``"auto"`` (the
            default) judges from what was said and how whether the user has
            finished, and tells the user's turns from noise and other voices.
            ``"vad"`` ends a turn after ``silence_timeout_ms`` of silence.
            ``"manual"`` ends a turn only on :meth:`FishAudioSTTService.force_end_turn`
            or, with ``vad_force_turn_endpoint``, on each VAD stop. Updated on the
            open session; a session that starts in ``"vad"`` or ``"manual"`` and
            switches to ``"auto"`` treats every later turn as the user's.
        silence_timeout_ms: Silence that ends a turn. In ``"auto"`` mode a
            safety net (3000-60000, server default 5000); in ``"vad"`` mode the
            endpointing delay (300-60000, server default 700); in ``"manual"``
            mode a safety net (300-60000, server default 60000).
        text_format: How numbers, amounts and times are written:
            ``"spoken"`` (words, the server default) or ``"written"`` (digits
            and symbols). ``transcribe-stream-2`` only, set for the whole
            session, so a change reconnects.
        tag_audio_events: Whether to keep sound tags such as ``[laughter]`` in
            transcripts. Defaults to False.
        diarize: Whether Fish Audio labels speakers. Labeled transcripts are
            split by speaker, with each part's ``speaker`` set. Defaults to False.
    """

    keyterms: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_detection_mode: FishAudioSTTTurnDetectionMode | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    silence_timeout_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_format: FishAudioSTTTextFormat | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    tag_audio_events: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarize: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _Turn:
    """A turn Fish Audio opened, from ``speech_start`` to ``committed``."""

    index: int
    start_ms: int
    user_turn: bool = False
    draft: str = ""


class FishAudioSTTService(EagerEndOfTurnSTTServiceMixin, WebsocketSTTService):
    """Speech-to-text service using Fish Audio's realtime ASR.

    Fish Audio transcribes the audio turn by turn and, in its default
    ``"auto"`` mode, decides when the user has finished speaking and which
    sounds are not the user taking their turn (noise, a TV, other people).
    This service pushes a :class:`ProposedUserStartedSpeakingFrame` once the
    user's turn has words (the server judged that the user took the turn, and
    a draft or the commit holds more than sound tags), an
    :class:`InterimTranscriptionFrame` for each draft, and on commit a final
    :class:`TranscriptionFrame` followed by a
    :class:`ProposedUserStoppedSpeakingFrame`. It recommends
    :class:`~pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies`, so
    the pipeline needs no local VAD or turn analyzer. Turns that are not the
    user's, and user turns without words, are dropped. The service doesn't
    tell Fish Audio when the bot is speaking, which Fish Audio needs to tell an
    acknowledgement from the user taking the turn; a turn Fish Audio does
    commit as a backchannel is pushed as a :class:`UserBackchannelFrame`.

    With ``enable_eager_end_of_turn``, the server's predicted end of turn is
    pushed as an :class:`EagerTranscriptionFrame` and withdrawn when the user
    goes on, so a response can be prepared ahead of the commit.

    Without a local VAD, the service also pushes ``VADUserStarted/
    StoppedSpeakingFrame`` placed at the speech's real start and end (from the
    turn's start and its last word's end), so observers that time the
    conversation from them, such as
    :class:`~pipecat.observers.user_bot_latency_observer.UserBotLatencyObserver`,
    keep working. It stops as soon as the pipeline announces a VAD of its own.

    With ``vad_force_turn_endpoint=True`` the pipeline's own user turn
    strategies decide instead: each ``VADUserStoppedSpeakingFrame`` asks Fish
    Audio to commit the turn, which gives a final transcript a moment after
    the speech ends.

    The bot's replies are sent to the session as conversation context, which
    helps recognition and the user-turn judgement. A session the server ends
    (a restart, or the one-hour limit) is replaced by a new one, seeded with
    the conversation so far.

    Event handlers available (in addition to the base ``on_connected`` /
    ``on_disconnected`` / ``on_connection_error``):

    - on_session_started(service, data): the server started a session; ``data``
      is its ``session_started`` message, whose ``config.model`` names the model
    - on_turn_committed(service, data): the server committed a turn, the user's
      or not; ``data`` is its ``committed`` message

    Example::

        stt = FishAudioSTTService(
            api_key=os.getenv("FISH_API_KEY"),
            settings=FishAudioSTTService.Settings(model="transcribe-stream-2"),
        )
    """

    Settings = FishAudioSTTSettings
    _settings: Settings

    # How long to wait for ``session_started`` after the handshake.
    _CONNECTION_TIMEOUT = 10.0
    # How long to wait at stop for the server to commit the open turn and close.
    _DRAIN_TIMEOUT = 2.0

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.fish.audio/v1/asr/realtime",
        sample_rate: int | None = None,
        interim_results: bool = True,
        should_interrupt: bool = True,
        enable_eager_end_of_turn: bool = False,
        vad_force_turn_endpoint: bool = False,
        emit_vad_frames: bool | None = None,
        ttfs_p99_latency: float | None = FISH_TTFS_P99,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Fish Audio STT service.

        Args:
            api_key: Fish Audio API key.
            url: WebSocket URL of the realtime ASR endpoint.
            sample_rate: Audio sample rate in Hz. If ``None``, uses the
                pipeline's input sample rate.
            interim_results: Whether to push live drafts as
                :class:`InterimTranscriptionFrame`. Drafts are requested either
                way, since they let a turn start before it commits. Defaults to
                True.
            should_interrupt: Whether a user turn interrupts the bot. Passed
                along to the user turn strategies this service recommends; a
                user-supplied ``user_turn_strategies`` overrides it. Ignored
                with ``vad_force_turn_endpoint=True``.
            enable_eager_end_of_turn: Whether to answer the server's predicted
                end of turn ahead of the committed one. ``"auto"`` mode only.
                Off by default: it spends an inference on every prediction,
                including the ones the server withdraws.
            vad_force_turn_endpoint: Whether the pipeline's user turn
                strategies decide when user turns end, instead of Fish Audio.
                Each ``VADUserStoppedSpeakingFrame`` then commits Fish Audio's
                open turn. Defaults to False.
            emit_vad_frames: Whether to push ``VADUserStarted/StoppedSpeakingFrame``
                from Fish Audio's turns. ``None`` (the default) pushes them until
                the pipeline announces a VAD of its own, and never with
                ``vad_force_turn_endpoint``.
            ttfs_p99_latency: P99 latency from speech end to final transcript,
                in seconds. Only reported with ``vad_force_turn_endpoint=True``,
                where a VAD marks the end of speech.
            settings: Runtime-updatable settings, including the model
                (``"transcribe-stream-1"``, the default, or
                ``"transcribe-stream-2"``).
            **kwargs: Additional arguments passed to the parent
                :class:`WebsocketSTTService`.

        Raises:
            ValueError: If the settings are outside what Fish Audio accepts, or
                ``enable_eager_end_of_turn`` and ``vad_force_turn_endpoint`` are
                both set.
        """
        if vad_force_turn_endpoint and enable_eager_end_of_turn:
            raise ValueError(
                "enable_eager_end_of_turn is not supported with vad_force_turn_endpoint=True, "
                "where the pipeline's user turn strategies decide when the turn ends"
            )

        default_settings = self.Settings(
            model="transcribe-stream-1",
            language=None,
            keyterms=None,
            turn_detection_mode="auto",
            silence_timeout_ms=None,
            text_format=None,
            tag_audio_events=False,
            diarize=False,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        if default_settings.language is not None:
            logger.warning("Fish Audio STT detects the language of each turn; `language` is unused")
            default_settings.language = None
        mode = default_settings.turn_detection_mode
        _validate_turn_detection(mode, default_settings.silence_timeout_ms)
        if default_settings.text_format not in (None, "spoken", "written"):
            raise ValueError('text_format must be "spoken" or "written"')

        super().__init__(
            sample_rate=sample_rate,
            enable_eager_end_of_turn=enable_eager_end_of_turn,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=_KEEPALIVE_SECS,
            keepalive_interval=1.0,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._interim_results = interim_results
        self._should_interrupt = should_interrupt
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._emit_vad_frames = emit_vad_frames
        if vad_force_turn_endpoint and emit_vad_frames:
            logger.warning(
                f"{self}: emit_vad_frames is ignored with vad_force_turn_endpoint=True, "
                "where the pipeline's VAD marks speech"
            )
        # Whether the pipeline announced a VAD of its own.
        self._pipeline_vad_seen = False

        self._session_confirmed = asyncio.Event()
        # Whether a disconnect lets the server commit the open turn first.
        self._drain_on_disconnect = True
        self._session_id: str | None = None
        self._request_id: str | None = None
        self._receive_task: asyncio.Task | None = None
        # Set once the server refused `text_format` on a model without it, so
        # later sessions on that model don't send it again.
        self._text_format_refused = False
        # The turn detection mode the server confirmed for the session.
        self._server_mode: str | None = None
        # Settings changed while the session was unconfirmed, sent once it starts.
        self._unsent_configure_fields: set[str] = set()
        self._unsent_rollback: dict[str, Any] = {}
        # The previous values of the fields each unanswered `configure` changed,
        # oldest first, restored if the server refuses it.
        self._configure_rollbacks: deque[dict[str, Any]] = deque()
        # A reconnect waiting for the open turn to commit.
        self._reconnect_after_turn = False

        # Audio waiting to make up a message of at least _MIN_CHUNK_SECS.
        self._pending_audio = bytearray()
        # Where this connection's audio stands, in ms since its first byte, and
        # (end_ms, wall time) for each recent message, to place audio positions
        # in wall-clock time.
        self._audio_sent_ms = 0.0
        self._sent_chunks: deque[tuple[float, float]] = deque(maxlen=3000)

        self._turn: _Turn | None = None
        # Whether the user's turn was announced to the pipeline and not yet ended.
        self._user_turn_open = False
        # Whether a VADUserStartedSpeakingFrame awaits its stop.
        self._vad_started = False
        self._language: Language | None = None
        # The conversation so far, oldest first, which seeds each new session.
        self._history: deque[dict[str, str]] = deque(maxlen=_MAX_HISTORY_ITEMS)

        self._register_event_handler("on_session_started")
        self._register_event_handler("on_turn_committed")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Fish Audio STT supports metrics generation.
        """
        return True

    @property
    def supports_ttfs(self) -> bool:
        """Whether TTFS applies: only when a VAD, not Fish Audio, marks the end of speech."""
        return self._vad_force_turn_endpoint

    @property
    def send_sample_rate(self) -> int:
        """The pipeline's rate when Fish Audio accepts it, otherwise 16 kHz."""
        if self.sample_rate in _SAMPLE_RATES:
            return self.sample_rate
        return _DEFAULT_SEND_SAMPLE_RATE

    @property
    def session_id(self) -> str | None:
        """The current Fish Audio session's ID, for support requests."""
        return self._session_id

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend turn strategies that leave turn detection to Fish Audio.

        Fish Audio proposes user turn starts and stops itself, so the user
        aggregator resolves those rather than running a local VAD or turn
        analyzer. With ``enable_eager_end_of_turn``, the recommendation also
        answers the server's predicted end of turn. Applied unless the user
        passed their own ``user_turn_strategies``. Nothing is recommended with
        ``vad_force_turn_endpoint=True``.
        """
        frame = super().service_metadata_frame()
        if not self._vad_force_turn_endpoint:
            eager_strategies = self.eager_user_turn_strategies(
                enable_interruptions=self._should_interrupt
            )
            if eager_strategies is not None:
                frame.user_turn_strategies = eager_strategies
            else:
                frame.user_turn_strategies = ExternalUserTurnStrategies(
                    enable_interruptions=self._should_interrupt
                )
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service, closing the connection without waiting for the server.

        Args:
            frame: The cancel frame.
        """
        self._drain_on_disconnect = False
        await super().cancel(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, committing the open turn on each VAD stop when asked to.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, SpeechControlParamsFrame) and frame.vad_params is not None:
            if not self._pipeline_vad_seen:
                logger.debug(f"{self}: the pipeline has a VAD, not pushing VAD frames")
            self._pipeline_vad_seen = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._vad_force_turn_endpoint:
            await self.force_end_turn()

    async def force_end_turn(self):
        """Ask Fish Audio to commit the open turn now, e.g. on a push-to-talk release.

        The committed turn is always the user's; with
        ``turn_detection_mode="manual"`` this is what ends turns. The server
        ignores it when no turn is open, and the service then waits for no
        finalized transcript.
        """
        if not self._is_connection_open():
            return
        await self._flush_pending_audio()
        if self._turn is not None:
            self.request_finalize()
        await self._send({"type": "finalize"})

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta.

        The model and ``text_format`` are fixed for a session, so a change to
        either opens a new session, once the open turn has committed. Turn
        detection, keyterms, sound tags and diarization are updated on the
        open session with ``configure``; if the server refuses one, they go
        back to their previous values.

        A mode change without ``silence_timeout_ms`` returns it to the new
        mode's default, as the server does.

        Args:
            delta: A :class:`STTSettings` (or :class:`FishAudioSTTSettings`) delta.

        Returns:
            Dict mapping changed field names to their previous values.

        Raises:
            ValueError: If the resulting turn detection mode and
                ``silence_timeout_ms`` don't go together, or ``text_format``
                is neither ``"spoken"`` nor ``"written"``.
        """
        if isinstance(delta, FishAudioSTTSettings):
            self._prepare_turn_detection_delta(delta)
            if is_given(delta.text_format) and delta.text_format not in (
                None,
                "spoken",
                "written",
            ):
                raise ValueError('text_format must be "spoken" or "written"')

        changed = await super()._update_settings(delta)

        if "language" in changed:
            logger.warning(f"{self}: Fish Audio detects the language of each turn; ignoring it")
            self._settings.language = None

        if "text_format" in changed or "model" in changed:
            self._text_format_refused = False

        session_bound = {"model", "text_format"} & changed.keys()
        configurable = {
            "keyterms",
            "turn_detection_mode",
            "silence_timeout_ms",
            "tag_audio_events",
            "diarize",
        } & changed.keys()

        if session_bound:
            await self._request_session_reconnect()
        elif configurable:
            rollback = {}
            for name in configurable:
                rollback[name] = changed[name]
            await self._send_configure(configurable, rollback)

        return changed

    def _prepare_turn_detection_delta(self, delta: FishAudioSTTSettings):
        """Validate the turn detection a delta leads to, clearing a timeout a mode change voids."""
        mode_given = is_given(delta.turn_detection_mode)
        timeout_given = is_given(delta.silence_timeout_ms)
        if not mode_given and not timeout_given:
            return
        mode = self._settings.turn_detection_mode
        if mode_given:
            mode = delta.turn_detection_mode
        timeout = self._settings.silence_timeout_ms
        if timeout_given:
            timeout = delta.silence_timeout_ms
        elif mode_given and mode != self._settings.turn_detection_mode:
            # The server resets the timeout to the new mode's default.
            timeout = None
            delta.silence_timeout_ms = None
        _validate_turn_detection(mode, timeout)

    async def _send_configure(self, fields: set[str], rollback: dict[str, Any]):
        """Send changed settings to the session.

        A session not yet confirmed gets them once it starts; with no
        connection open, the next session's config carries them.
        """
        if not self._is_connection_open():
            return
        if not self._session_confirmed.is_set():
            self._unsent_configure_fields.update(fields)
            for name, value in rollback.items():
                self._unsent_rollback.setdefault(name, value)
            return
        message: dict[str, Any] = {"type": "configure"}
        if "turn_detection_mode" in fields or "silence_timeout_ms" in fields:
            message["turn_detection"] = self._turn_detection_config()
        results: dict[str, Any] = {}
        if "tag_audio_events" in fields:
            results["tag_audio_events"] = bool(self._settings.tag_audio_events)
        if "diarize" in fields:
            results["diarize"] = bool(self._settings.diarize)
        if results:
            message["results"] = results
        if "keyterms" in fields:
            # Replaces the session's whole list.
            message["context"] = {"terms": _sanitize_terms(self._settings.keyterms)}
        self._configure_rollbacks.append(rollback)
        await self._send(message)

    def _turn_detection_config(self) -> dict[str, Any]:
        """The session's turn detection, with the timeout always explicit."""
        mode = self._settings.turn_detection_mode
        turn_detection: dict[str, Any] = {"mode": mode}
        if mode == "auto" and self.eager_end_of_turn_enabled:
            turn_detection["eager"] = True
        timeout = self._settings.silence_timeout_ms
        if timeout is None:
            timeout = _DEFAULT_SILENCE_TIMEOUT_MS[mode]
        turn_detection["silence_timeout_ms"] = timeout
        return turn_detection

    def _roll_back_configure(self, rollback: dict[str, Any], message: str | None):
        """Restore the settings a refused ``configure`` had changed."""
        logger.warning(
            f"{self}: Fish Audio refused a settings update, keeping the old one: {message}"
        )
        for name, value in rollback.items():
            setattr(self._settings, name, value)

    def _session_config(self) -> dict[str, Any]:
        """The first message of a session: audio format, turn detection, results and context."""
        mode = self._settings.turn_detection_mode
        turn_detection: dict[str, Any] = {"mode": mode}
        if mode == "auto" and self.eager_end_of_turn_enabled:
            turn_detection["eager"] = True
        if self._settings.silence_timeout_ms is not None:
            turn_detection["silence_timeout_ms"] = self._settings.silence_timeout_ms

        results: dict[str, Any] = {
            # A turn is announced once a draft has words, so drafts are always on.
            "drafts": True,
            # Word timings place the end of speech, which the commit lands after.
            "ignore_timestamps": False,
            "diarize": bool(self._settings.diarize),
            "tag_audio_events": bool(self._settings.tag_audio_events),
        }
        if self._settings.text_format is not None and not self._text_format_refused:
            results["text_format"] = self._settings.text_format

        config: dict[str, Any] = {
            "type": "config",
            "protocol_version": 1,
            "audio": {
                "format": "pcm_s16le",
                "sample_rate": self.send_sample_rate,
                "channels": 1,
            },
            "turn_detection": turn_detection,
            "results": results,
        }

        context: dict[str, Any] = {}
        if self._history:
            context["history"] = list(self._history)
        terms = _sanitize_terms(self._settings.keyterms)
        if terms:
            context["terms"] = terms
        if context:
            config["context"] = context
        return config

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect and start the receive loop, which owns reconnection for the session."""
        await super()._connect()
        self._drain_on_disconnect = True
        await self._connect_websocket()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """End the session, letting the server commit the open turn before it closes."""
        await super()._disconnect()
        try:
            audio_done_sent = False
            if self._is_session_open() and self._drain_on_disconnect:
                try:
                    await self._flush_pending_audio()
                    await self._send({"type": "audio_done"})
                    audio_done_sent = True
                except Exception as e:
                    logger.debug(f"{self}: could not send audio_done: {e}")
            if self._receive_task:
                if audio_done_sent:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self._receive_task), timeout=self._DRAIN_TIMEOUT
                        )
                    except TimeoutError:
                        logger.debug(f"{self}: timed out waiting for the server to close")
                    except Exception:
                        pass
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open a connection and send the session's config.

        The server confirms the session with ``session_started``, which the
        receive loop waits for.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            # A turn lives in one session; one the last session left open ends here.
            await self._end_turn_cut_short()
            self._session_confirmed.clear()
            self._session_id = None
            self._request_id = None
            self._server_mode = self._settings.turn_detection_mode
            self._unsent_configure_fields.clear()
            self._unsent_rollback = {}
            self._configure_rollbacks.clear()
            self._audio_sent_ms = 0.0
            self._sent_chunks.clear()

            model = self._settings.model
            headers = {"Authorization": f"Bearer {self._api_key}"}
            if model:
                headers["model"] = model
            logger.debug(f"{self}: connecting to Fish Audio realtime ASR ({model})")
            websocket = await self._websocket_connect(self._url, additional_headers=headers)
            self._websocket = websocket
            config = self._session_config()
            await websocket.send(ormsgpack.packb(config))
            # The server reads audio sent before session_started once the session starts.
            await self._flush_pending_audio()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the connection and reset its state."""
        try:
            self._session_confirmed.clear()
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Fish Audio realtime ASR")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _is_connection_open(self) -> bool:
        """Whether a connection is open, its session confirmed or not."""
        if not self._websocket:
            return False
        return self._websocket.state is State.OPEN

    def _is_session_open(self) -> bool:
        """Whether a confirmed session is open to take messages."""
        if not self._is_connection_open():
            return False
        return self._session_confirmed.is_set()

    async def _request_session_reconnect(self):
        """Open a new session, once the open turn has committed."""
        if self._turn is not None:
            self._reconnect_after_turn = True
            return
        await self._request_reconnect()

    def _maybe_reconnect_after_turn(self):
        """Run a reconnect that waited for the turn that just committed."""
        if not self._reconnect_after_turn:
            return
        self._reconnect_after_turn = False
        # Reconnecting ends this receive loop, so it runs on a task of its own.
        self.create_task(self._request_reconnect(), name="fish_stt_reconnect")

    def _is_keepalive_ready(self) -> bool:
        """Keepalives go to a confirmed session only."""
        return self._is_session_open()

    async def _send_keepalive(self, silence: bytes):
        """Keep the idle session open with a ``keepalive`` message rather than silence."""
        await self._send({"type": "keepalive"})

    async def _send(self, message: dict[str, Any]):
        """Send one MessagePack message on the open connection."""
        websocket = self._websocket
        if not websocket or websocket.state is not State.OPEN:
            return
        payload = ormsgpack.packb(message)
        await websocket.send(payload)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Fish Audio in messages of at least 20 ms.

        Transcripts are pushed by the receive task. While no connection is
        open, the last few seconds of audio are kept and sent as soon as the
        next connection opens.

        Args:
            audio: 16-bit mono PCM audio at :attr:`send_sample_rate`.

        Yields:
            Frame: None, as transcripts are pushed by the receive task.
        """
        self._pending_audio.extend(audio)
        if not self._is_connection_open():
            max_bytes = int(self.send_sample_rate * _MAX_BUFFERED_SECS) * 2
            excess = len(self._pending_audio) - max_bytes
            if excess > 0:
                del self._pending_audio[:excess]
            yield None
            return
        min_bytes = int(self.send_sample_rate * _MIN_CHUNK_SECS) * 2
        if len(self._pending_audio) >= min_bytes:
            try:
                await self._flush_pending_audio()
            except Exception as e:
                logger.warning(f"{self}: failed to send audio: {e}")
        yield None

    async def _flush_pending_audio(self):
        """Send the audio waiting to make up a message."""
        if not self._pending_audio:
            return
        chunk = bytes(self._pending_audio)
        self._pending_audio = bytearray()
        await self._send({"type": "audio", "data": chunk})
        chunk_ms = len(chunk) / 2 / self.send_sample_rate * 1000
        self._audio_sent_ms += chunk_ms
        self._sent_chunks.append((self._audio_sent_ms, time.time()))

    def _wall_time(self, audio_ms: float | None) -> float | None:
        """Wall-clock time a position in this connection's audio was captured.

        Each message's last sample was captured about when it was sent, so a
        position is placed back from the first message sent after it.
        """
        if audio_ms is None or not self._sent_chunks:
            return None
        for chunk_end_ms, sent_at in self._sent_chunks:
            if chunk_end_ms >= audio_ms:
                return sent_at - (chunk_end_ms - audio_ms) / 1000
        last_end_ms = self._sent_chunks[-1][0]
        last_sent_at = self._sent_chunks[-1][1]
        return last_sent_at + (audio_ms - last_end_ms) / 1000

    # ------------------------------------------------------------------
    # Receiving
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Wait for ``session_started``, then handle the session's messages."""
        messages = aiter(self._get_websocket())
        confirmed = await self._receive_until_confirmed(
            messages, self._session_confirmed, self._process_message, self._CONNECTION_TIMEOUT
        )
        if not confirmed:
            return

        await self._call_event_handler("on_connected")

        try:
            async for message in messages:
                await self._process_message(message)
        finally:
            # A turn lives in one session, so it ends with it, unless the service is stopping.
            if not self._disconnecting:
                await self._end_turn_cut_short()

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _process_message(self, message: str | bytes):
        """Decode and handle one server message."""
        if not isinstance(message, bytes):
            logger.warning(f"{self}: unexpected text message from Fish Audio")
            return
        try:
            data = ormsgpack.unpackb(message)
        except Exception:
            logger.exception(f"{self}: failed to decode a Fish Audio message")
            return
        if not isinstance(data, dict):
            return
        try:
            await self._handle_message(data)
        except _TextFormatRefusedError:
            # Already logged; the receive loop reconnects.
            raise
        except FishAudioSTTRequestError as e:
            await self.push_error(error_msg=str(e), exception=e)
            raise
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _handle_message(self, data: dict[str, Any]):
        message_type = data.get("type")
        if message_type == "session_started":
            await self._handle_session_started(data)
        elif message_type == "speech_start":
            await self._handle_speech_start(data)
        elif message_type == "user_turn_taken":
            await self._handle_user_turn_taken(data)
        elif message_type == "draft":
            await self._handle_draft(data)
        elif message_type == "eager_end_of_turn":
            await self._handle_eager_end_of_turn(data)
        elif message_type == "turn_resumed":
            await self._cancel_eager_end_of_turn()
        elif message_type == "committed":
            await self._handle_committed(data)
        elif message_type == "user_backchannel":
            logger.debug(f"{self}: user backchannel at {data.get('audio_ms')} ms")
        elif message_type == "configured":
            await self._handle_configured(data)
        elif message_type == "error":
            await self._handle_server_error(data)
        elif message_type == "session_done":
            logger.debug(f"{self}: session done: {data.get('stats')}")
        # progress, vad, eot_prob and unknown types need no handling.

    async def _handle_session_started(self, data: dict[str, Any]):
        self._session_id = data.get("session_id")
        self._request_id = data.get("request_id")
        config = data.get("config") or {}
        served_model = config.get("model")
        requested_model = self._settings.model
        logger.info(
            f"{self}: Fish Audio session {self._session_id} started "
            f"(model {served_model or requested_model}, request_id {self._request_id})"
        )
        if served_model and requested_model and served_model != requested_model:
            logger.warning(
                f"{self}: asked for {requested_model} but Fish Audio serves this session "
                f"with {served_model}"
            )
        self._read_server_mode(config)
        self._session_confirmed.set()
        await self._call_event_handler("on_session_started", data)
        if self._unsent_configure_fields:
            fields = set(self._unsent_configure_fields)
            rollback = dict(self._unsent_rollback)
            self._unsent_configure_fields.clear()
            self._unsent_rollback = {}
            await self._send_configure(fields, rollback)

    async def _handle_configured(self, data: dict[str, Any]):
        rollback: dict[str, Any] = {}
        if self._configure_rollbacks:
            rollback = self._configure_rollbacks.popleft()
        if not data.get("ok", True):
            self._roll_back_configure(rollback, data.get("message"))
        self._read_server_mode(data.get("config") or {})

    def _read_server_mode(self, config: dict[str, Any]):
        """Keep the turn detection mode the server reports as in effect."""
        turn_detection = config.get("turn_detection") or {}
        mode = turn_detection.get("mode")
        if mode:
            self._server_mode = mode

    async def _handle_server_error(self, data: dict[str, Any]):
        error_type = data.get("error_type")
        message = data.get("message") or "unknown error"
        error_text = f"Fish Audio STT error [{error_type}]: {message}"
        if error_type in _INVALID_SETTINGS_ERROR_TYPES:
            sent_text_format = self._settings.text_format is not None
            if sent_text_format and "text_format" in message and not self._text_format_refused:
                # The session is on a model without text_format, such as
                # transcribe-stream-1 when transcribe-stream-2 isn't available.
                # A new session without it can still transcribe.
                logger.warning(
                    f"{self}: Fish Audio refused text_format (the session's model doesn't "
                    f"take it); reconnecting without it: {message}"
                )
                self._text_format_refused = True
                raise _TextFormatRefusedError(error_text)
            raise FishAudioSTTRequestError(error_text, status_code=data.get("code", 400))
        if data.get("recoverable"):
            # The session goes on; server_restarting closes it with 1012 and the
            # receive loop opens a new one.
            logger.warning(f"{self}: {error_text} (request_id {data.get('request_id')})")
            return
        logger.warning(f"{self}: {error_text}; the session will close")

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def _current_turn(self, data: dict[str, Any]) -> _Turn | None:
        turn = self._turn
        if turn is None or turn.index != data.get("turn_index"):
            return None
        return turn

    async def _handle_speech_start(self, data: dict[str, Any]):
        turn_index = data.get("turn_index", 0)
        start_ms = data.get("audio_ms", 0)
        turn = _Turn(index=turn_index, start_ms=start_ms)
        self._turn = turn
        # Outside auto mode every turn is the user's; in auto mode a turn can
        # be noise or another voice until user_turn_taken.
        if self._server_mode != "auto":
            turn.user_turn = True

    async def _handle_user_turn_taken(self, data: dict[str, Any]):
        turn = self._current_turn(data)
        if turn is None:
            turn_index = data.get("turn_index", 0)
            start_ms = data.get("audio_ms", 0)
            turn = _Turn(index=turn_index, start_ms=start_ms)
            self._turn = turn
        turn.user_turn = True
        await self._maybe_start_user_turn(turn, turn.draft)

    async def _maybe_start_user_turn(self, turn: _Turn, text: str):
        """Announce the user's turn once it has words.

        A turn of sounds alone, such as a laugh, commits with no words to
        answer, so it neither interrupts the bot nor opens a turn the
        pipeline would wait on.
        """
        if self._user_turn_open or not turn.user_turn:
            return
        if not _words_only(text):
            return
        await self._start_user_turn(turn.start_ms)
        if turn.draft and self._interim_results:
            await self._push_interim(turn.draft, None)

    async def _start_user_turn(self, start_ms: int):
        self._user_turn_open = True
        logger.debug(f"{self}: user started speaking")
        await self._push_vad_started(start_ms)
        if not self._vad_force_turn_endpoint:
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _handle_draft(self, data: dict[str, Any]):
        turn = self._current_turn(data)
        if turn is None:
            return
        text = (data.get("text") or "").strip()
        turn.draft = text
        if not turn.user_turn:
            return
        if not self._user_turn_open:
            # Pushes this draft once the turn starts.
            await self._maybe_start_user_turn(turn, text)
            return
        if text and self._interim_results:
            await self._push_interim(text, data)

    async def _push_interim(self, text: str, data: dict[str, Any] | None):
        language = self._read_language(data)
        clean_text = _strip_speaker_tags(text)
        if not clean_text:
            return
        await self.push_frame(
            InterimTranscriptionFrame(
                clean_text, self._user_id, time_now_iso8601(), language, result=data
            )
        )

    async def _handle_eager_end_of_turn(self, data: dict[str, Any]):
        turn = self._current_turn(data)
        if turn is None or not turn.user_turn:
            return
        text = _strip_speaker_tags(data.get("text") or "")
        if not _words_only(text):
            return
        await self._maybe_start_user_turn(turn, text)
        language = self._read_language(data)
        await self._push_eager_end_of_turn(
            text, user_id=self._user_id, language=language, result=data
        )

    async def _handle_committed(self, data: dict[str, Any]):
        if self._current_turn(data) is not None:
            self._turn = None
        await self._call_event_handler("on_turn_committed", data)

        text = (data.get("text") or "").strip()
        # A missing user_turn counts as the user's.
        user_turn = data.get("user_turn", True)
        if user_turn:
            await self._handle_user_commit(data, text)
        else:
            await self._handle_non_user_turn(data, text)
        self._maybe_reconnect_after_turn()

    async def _handle_user_commit(self, data: dict[str, Any], text: str):
        reason = data.get("reason")
        if reason == "finalize":
            self.confirm_finalize()
        self._clear_eager_end_of_turn()

        words = _words_only(text)
        if not self._user_turn_open and words:
            # A turn the server ended before its drafts had words, e.g. on finalize.
            await self._start_user_turn(data.get("audio_start_ms", 0))
        if not self._user_turn_open:
            logger.debug(f"{self}: dropping a user turn without words ({reason=}): [{text}]")
            await self._push_transcription_finalized()
            return
        self._user_turn_open = False

        logger.debug(f"{self}: user stopped speaking ({reason=}): [{text}]")
        await self._push_vad_stopped(data)

        language = self._read_language(data)
        if words:
            # Report usage before the transcript, for the STT span it closes.
            await self.emit_stt_usage_metrics()
            runs = self._speaker_runs_of(text)
            finalized = not self._vad_force_turn_endpoint
            await self._push_transcription_runs(
                runs, language=language, result=data, finalized=finalized
            )
            spoken_text = _strip_speaker_tags(text)
            await self._handle_transcription(spoken_text, True, language)
            self._append_history("user", spoken_text)
        else:
            # The drafts had words the final pass didn't keep. The pipeline's
            # turn stop timeout ends the turn, as it does any turn without text.
            await self._push_transcription_finalized()

        if not self._vad_force_turn_endpoint:
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_non_user_turn(self, data: dict[str, Any], text: str):
        """Keep a turn that isn't the user's out of the conversation."""
        if self._user_turn_open:
            # The turn was announced as the user's under a mode the server didn't apply.
            self._user_turn_open = False
            await self._push_vad_stopped(data)
            if not self._vad_force_turn_endpoint:
                await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        turn_kind = data.get("turn_kind")
        if turn_kind == "backchannel" and _words_only(text):
            await self.push_frame(UserBackchannelFrame(text, self._user_id, time_now_iso8601()))
            return
        logger.debug(f"{self}: dropping a turn that isn't the user's ({turn_kind=}): [{text}]")

    def _speaker_runs_of(self, text: str) -> list[tuple[TranscriptionSpeaker | None, str]]:
        """Split a committed transcript into speaker runs, or one run without diarization."""
        if not self._settings.diarize:
            return [(None, _strip_speaker_tags(text))]
        runs: list[tuple[TranscriptionSpeaker | None, str]] = []
        for label, run_text in _split_speakers(text):
            speaker = self._speaker(label)
            formatted = self._apply_speaker_format(run_text, speaker)
            runs.append((speaker, formatted))
        if not runs:
            return [(None, _strip_speaker_tags(text))]
        return runs

    async def _end_turn_cut_short(self):
        """End a user turn whose session closed before the server committed it.

        The server commits the open turn before it closes a session itself, so
        this only happens when a connection drops. The draft so far is pushed,
        unfinalized, and the turn stop is proposed.
        """
        turn = self._turn
        self._turn = None
        if not self._user_turn_open:
            return
        logger.debug(f"{self}: the session closed mid-turn, ending the turn")
        self._user_turn_open = False
        self._clear_eager_end_of_turn()
        await self._push_vad_stopped({})
        if turn is not None and _words_only(turn.draft):
            text = _strip_speaker_tags(turn.draft)
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(text, self._user_id, time_now_iso8601(), self._language)
            )
            self._append_history("user", text)
        if not self._vad_force_turn_endpoint:
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    def _read_language(self, data: dict[str, Any] | None) -> Language | None:
        """The message's language, or the last one seen when it's too short to tell."""
        if data is None:
            return self._language
        code = data.get("language_code")
        if not code:
            return self._language
        try:
            self._language = Language(code)
        except ValueError:
            logger.debug(f"{self}: no Language for language code {code!r}")
        return self._language

    # ------------------------------------------------------------------
    # VAD frames for a pipeline without its own VAD
    # ------------------------------------------------------------------

    def _should_push_vad_frames(self) -> bool:
        if self._vad_force_turn_endpoint:
            return False
        if self._emit_vad_frames is not None:
            return self._emit_vad_frames
        return not self._pipeline_vad_seen

    async def _push_vad_started(self, start_ms: int):
        """Mark the start of the user's speech, at the turn's start."""
        if not self._should_push_vad_frames():
            return
        now = time.time()
        started_at = self._wall_time(start_ms)
        start_secs = 0.0
        if started_at is not None:
            start_secs = max(0.0, now - started_at)
        self._vad_started = True
        await self.push_frame(VADUserStartedSpeakingFrame(start_secs=start_secs, timestamp=now))

    async def _push_vad_stopped(self, data: dict[str, Any]):
        """Mark the end of the user's speech, for a turn whose start was marked.

        A commit lands after the silence that ended the turn, so the frame's
        ``stop_secs`` holds that wait, and ``timestamp - stop_secs`` is the end
        of speech: the end of the turn's last word, or, for a commit without
        word timings, the end of the turn.
        """
        if not self._vad_started:
            return
        self._vad_started = False
        ended_ms = data.get("audio_end_ms")
        words = data.get("words") or []
        if words:
            ended_ms = words[-1].get("end_ms")
        now = time.time()
        ended_at = self._wall_time(ended_ms)
        stop_secs = 0.0
        if ended_at is not None:
            stop_secs = max(0.0, now - ended_at)
        await self.push_frame(VADUserStoppedSpeakingFrame(stop_secs=stop_secs, timestamp=now))

    # ------------------------------------------------------------------
    # Conversation context
    # ------------------------------------------------------------------

    def _append_history(self, role: str, text: str):
        item = {"role": role, "text": text[-_MAX_HISTORY_CHARS:]}
        self._history.append(item)

    async def _process_assistant_turn(self, text: str):
        """Send the bot's reply to Fish Audio as conversation context.

        Fish Audio adds committed user turns to its context itself; the bot's
        replies help it recognize follow-ups and judge the next turn.

        Args:
            text: The assistant's aggregated spoken text for this turn.
        """
        clean_text = text.strip()
        if not clean_text:
            return
        self._append_history("assistant", clean_text)
        item = self._history[-1]
        if not self._is_session_open():
            return
        try:
            await self._send({"type": "context_add", "items": [item]})
        except Exception as e:
            logger.debug(f"{self}: could not send conversation context: {e}")
