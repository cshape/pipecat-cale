#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio text-to-speech service implementation.

This module provides integration with Fish Audio's real-time TTS WebSocket API
for streaming text-to-speech synthesis with customizable voice parameters.
"""

import asyncio
import re
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import (
    FishAudioDrama3ExpressionAdapter,
    FishAudioS2ExpressionAdapter,
)
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

try:
    import ormsgpack
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Fish Audio, you need to `uv add "pipecat-ai[fish]"`.')
    raise ImportError(f"Missing module: {e}") from e

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]

# How long the latest chunk's word alignment must go without an update before
# its last word is reported. Fish sends a growing snapshot of each chunk's
# alignment; a word is reported once a later word follows it, and a chunk's
# last word once the next chunk starts or after this long without a message.
ALIGNMENT_SETTLE_SECS = 0.25

# How many recent chunks keep their alignment state. Fish repeats a finished
# chunk's alignment on the first audio of the next reply, so a chunk is kept a
# while after it ends, to tell that repeat from a new chunk.
_KEPT_CHUNKS = 8

# A bracketed tag, such as "[excited]" or "[short pause]", which Fish Audio
# performs and leaves out of its word alignment; angle-bracket markup, such as
# "<whisper>", which word tracking skips by itself; and a word.
_TEXT_TOKEN_RE = re.compile(r"\[[^\[\]]*\]|<[^<>]*>|[^\s\[<]+")


@dataclass
class FishAudioTTSSettings(TTSSettings):
    """Settings for FishAudioTTSService.

    Parameters:
        language: BCP-47 tag of the text, such as ``en`` or ``es-MX``. Fish Audio uses it
            only to spell out numbers, dates, units and symbols; it does not change the
            voice. Unset detects the language of each chunk.
        latency: Latency mode ("normal" or "balanced"). Defaults to "balanced".
        normalize: Whether to normalize audio output. Defaults to True.
        temperature: Controls randomness in speech generation (0.0-1.0).
        top_p: Controls diversity via nucleus sampling (0.0-1.0).
        prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
        prosody_volume: Volume adjustment in dB (-20 to 20). Defaults to 0.
        prosody_normalize_loudness: Whether to normalize output loudness for a more
            consistent perceived volume. Applies to the S2 family and drama-3.
        chunk_length: Text segment size for processing (100-300).
        min_chunk_length: Minimum characters before splitting into a new chunk (0-100).
        condition_on_previous_chunks: Whether to use previously generated audio as
            context, for voice consistency across chunks.
    """

    latency: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    normalize: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_volume: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_normalize_loudness: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    chunk_length: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_chunk_length: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    condition_on_previous_chunks: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> Self:
        """Construct settings from a plain dict, destructuring legacy nested ``prosody``."""
        flat = dict(settings)
        nested = flat.pop("prosody", None)
        if isinstance(nested, dict):
            # Keys missing from the nested dict stay unset in the delta.
            for key, name in (
                ("speed", "prosody_speed"),
                ("volume", "prosody_volume"),
                ("normalize_loudness", "prosody_normalize_loudness"),
            ):
                if key in nested:
                    flat.setdefault(name, nested[key])
        return super().from_mapping(flat)


class FishAudioTTSService(InterruptibleTTSService):
    """Fish Audio text-to-speech service with WebSocket streaming.

    Provides real-time text-to-speech synthesis using Fish Audio's WebSocket API.
    Supports various audio formats, customizable prosody controls, and streaming
    audio generation with interruption handling.

    Uses the ``s2.1-pro`` model by default. Set ``Settings(model=...)`` to use
    another Fish Audio model, such as ``s3-pro``, ``s2-pro`` or ``drama-3``.

    By default it streams from Fish Audio's timestamped endpoint, which reports
    when each word is spoken. The bot's words then reach the context as they are
    heard, so an interrupted reply keeps only what the user heard. Pass
    ``word_timestamps=False`` to use the endpoint without timestamps, which
    reports each sentence as spoken once its audio is generated.
    """

    Settings = FishAudioTTSSettings
    _settings: Settings

    @deprecated(
        "`FishAudioTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `FishAudioTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for Fish Audio TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=FishAudioTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for synthesis. Defaults to English.
            latency: Latency mode ("normal" or "balanced"). Defaults to "normal".
            normalize: Whether to normalize audio output. Defaults to True.
            prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
            prosody_volume: Volume adjustment in dB. Defaults to 0.
        """

        language: Language | None = Language.EN
        latency: str | None = "normal"  # "normal" or "balanced"
        normalize: bool | None = True
        prosody_speed: float | None = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: int | None = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        reference_id: str | None = None,  # This is the voice ID
        model_id: str | None = None,
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: int | None = None,
        word_timestamps: bool = True,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key for authentication.
            reference_id: Reference ID of the voice model to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            model_id: Specify which Fish Audio TTS model to use (e.g. "s2.1-pro" or "drama-3").

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            output_format: Audio output format. Defaults to "pcm".
            sample_rate: Audio sample rate. If None, uses default.
            word_timestamps: Whether to stream from Fish Audio's timestamped
                endpoint and report each word as it is spoken. Defaults to True.
            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=FishAudioTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent service.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="s2.1-pro",
            voice=None,
            language=None,
            latency="balanced",
            normalize=True,
            temperature=None,
            top_p=None,
            prosody_speed=1.0,
            prosody_volume=0,
            prosody_normalize_loudness=None,
            chunk_length=None,
            min_chunk_length=None,
            condition_on_previous_chunks=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if reference_id is not None:
            self._warn_init_param_moved_to_settings("reference_id", "voice")
            default_settings.voice = reference_id
        if model_id is not None:
            self._warn_init_param_moved_to_settings("model_id", "model")
            default_settings.model = model_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.latency is not None:
                    default_settings.latency = params.latency
                if params.normalize is not None:
                    default_settings.normalize = params.normalize
                if params.prosody_speed is not None:
                    default_settings.prosody_speed = params.prosody_speed
                if params.prosody_volume is not None:
                    default_settings.prosody_volume = params.prosody_volume

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=not word_timestamps,
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._word_timestamps = word_timestamps
        if word_timestamps:
            self._base_url = "wss://api.fish.audio/v1/tts/live/with-timestamp"
        else:
            self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._s2_expression_adapter = FishAudioS2ExpressionAdapter()
        self._drama3_expression_adapter = FishAudioDrama3ExpressionAdapter()
        self._websocket = None
        self._receive_task = None
        self._alignment_task = None

        # Word alignment state for the open Fish session. Fish times each text
        # chunk from the start of the session's audio; words are reported to
        # Pipecat relative to the first audio of their audio context, which is
        # where that context's first chunk starts.
        self._chunks: dict[int, _ChunkAlignment] = {}
        self._latest_chunk: int | None = None
        self._context_audio_start: dict[str, float] = {}
        self._context_text: dict[str, _SentText] = {}
        self._last_alignment_time = 0.0
        self._alignment_updated = asyncio.Event()

        # Init-only audio format config (not runtime-updatable).
        self._fish_sample_rate = 0  # Set in start()
        self._output_format = output_format

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Fish Audio service supports metrics generation.
        """
        return True

    def get_expression_adapter(self) -> BaseExpressionAdapter:
        """The adapter for the model: bracket tags on S2 and S3, Drama 3's directions on drama-3.

        Returns:
            The expression adapter for the current model.
        """
        model = self._settings.model
        if not isinstance(model, str):
            return super().get_expression_adapter()
        if model.startswith("s2") or model.startswith("s3"):
            return self._s2_expression_adapter
        if model.startswith("drama-3"):
            return self._drama3_expression_adapter
        return super().get_expression_adapter()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Any change to voice or model triggers a WebSocket reconnect.

        Args:
            delta: A :class:`TTSSettings` (or ``FishAudioTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._fish_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Fish Audio TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Fish Audio TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._word_timestamps and not self._alignment_task:
            self._alignment_task = self.create_task(self._alignment_settle_handler())

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._alignment_task:
            await self.cancel_task(self._alignment_task)
            self._alignment_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Fish Audio")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            model = assert_given(self._settings.model)
            if model is not None:
                headers["model"] = model
            # With latency="normal", Fish sends a long sentence's audio in a few
            # large messages, which can exceed the websockets default of 1 MiB.
            websocket = await self._websocket_connect(
                self._base_url, additional_headers=headers, max_size=16 * 1024 * 1024
            )
            self._websocket = websocket
            self._reset_alignment()

            prosody: dict[str, Any] = {
                "speed": self._settings.prosody_speed,
                "volume": self._settings.prosody_volume,
            }
            if self._settings.prosody_normalize_loudness is not None:
                prosody["normalize_loudness"] = self._settings.prosody_normalize_loudness

            # Send initial start message with ormsgpack
            request_settings: dict[str, Any] = {
                "sample_rate": self._fish_sample_rate,
                "latency": self._settings.latency,
                "format": self._output_format,
                "normalize": self._settings.normalize,
                "prosody": prosody,
                "reference_id": self._settings.voice,
            }
            # Fish applies its own defaults to whatever the request leaves out.
            for name in (
                "temperature",
                "top_p",
                "chunk_length",
                "min_chunk_length",
                "condition_on_previous_chunks",
            ):
                value = getattr(self._settings, name)
                if value is not None:
                    request_settings[name] = value

            if self._settings.language is not None:
                request_settings["language"] = str(self._settings.language)
            start_message = {"event": "start", "request": {"text": "", **request_settings}}
            await websocket.send(ormsgpack.packb(start_message))
            logger.debug("Sent start message to Fish Audio")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any buffered audio by sending a flush event to Fish Audio."""
        logger.trace(f"{self}: Flushing audio buffers")
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        flush_message = {"event": "flush"}
        await self._get_websocket().send(ormsgpack.packb(flush_message))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop all metrics when audio context is interrupted."""
        await self.stop_all_metrics()
        self._forget_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Forget the word alignment state of a context that finished playing."""
        self._forget_context(context_id)
        await super().on_audio_context_completed(context_id)

    def _forget_context(self, context_id: str):
        """Drop a context's alignment state; its words still to come are not reported."""
        for chunk in self._chunks.values():
            if chunk.context_id == context_id:
                chunk.closed = True
        self._context_audio_start.pop(context_id, None)
        self._context_text.pop(context_id, None)

    def _reset_alignment(self):
        """Start the alignment clock over, for a new Fish session."""
        self._chunks = {}
        self._latest_chunk = None
        self._context_audio_start = {}
        self._alignment_updated.clear()

    async def _handle_alignment(self, message: dict[str, Any], context_id: str | None):
        """Keep a chunk's latest alignment snapshot and report the words that have settled."""
        alignment = message.get("alignment")
        if not alignment:
            return
        chunk_seq = message.get("chunk_seq")
        if not isinstance(chunk_seq, int):
            return
        chunk = self._chunks.get(chunk_seq)
        if chunk is None:
            latest = self._latest_chunk
            if latest is not None and chunk_seq < latest:
                return
            if context_id is None:
                return
            await self._finish_chunks_before(chunk_seq)
            chunk_offset = float(message.get("chunk_audio_offset_sec") or 0.0)
            chunk = _ChunkAlignment(context_id=context_id, offset=chunk_offset)
            self._chunks[chunk_seq] = chunk
            self._latest_chunk = chunk_seq
            if context_id not in self._context_audio_start:
                self._context_audio_start[context_id] = chunk_offset
            self._prune_chunks(chunk_seq)
        # A repeat of a finished chunk, which Fish sends with the next reply's
        # first audio, updates that chunk: its context is where it started.
        chunk.segments = _valid_segments(alignment.get("segments"))
        await self._report_chunk(chunk, final=False)
        self._last_alignment_time = time.monotonic()
        self._alignment_updated.set()

    async def _finish_chunks_before(self, chunk_seq: int):
        """Report the remaining words of the chunks before this one, which are done."""
        for seq in sorted(self._chunks):
            if seq < chunk_seq:
                await self._report_chunk(self._chunks[seq], final=True)

    async def _finish_all_chunks(self):
        """Report every word received, as the session finished or the socket closed."""
        for seq in sorted(self._chunks):
            await self._report_chunk(self._chunks[seq], final=True)

    def _prune_chunks(self, latest: int):
        for seq in list(self._chunks):
            if seq <= latest - _KEPT_CHUNKS:
                del self._chunks[seq]

    async def _report_chunk(self, chunk: "_ChunkAlignment", final: bool):
        """Report a chunk's words not reported yet, holding back the last one unless the chunk is done.

        Fish can still move a chunk's last word in a later snapshot, so a word
        is reported once a later word follows it. Each bracketed tag sent before
        a word is reported as a word of its own, at the end of the word before it,
        and a sent word Fish speaks in pieces is reported once, as it was sent.
        """
        if chunk.closed:
            return
        settled = len(chunk.segments)
        if not final:
            settled = settled - 1
        new_segments = []
        if settled > chunk.reported:
            new_segments = chunk.segments[chunk.reported : settled]
            chunk.reported = settled
        context_start = self._context_audio_start.get(chunk.context_id, chunk.offset)
        sent_text = self._context_text.get(chunk.context_id)
        word_times = []
        for segment in new_segments:
            start = chunk.offset + segment.start - context_start
            end = chunk.offset + segment.end - context_start
            # Fish times words to the hundredth of a second; rounding keeps
            # float noise from the offset arithmetic out of the timestamps.
            start = round(max(0.0, start), 3)
            end = round(max(start, end), 3)
            if sent_text is None:
                word_times.append((segment.text, start, end))
                continue
            placed = sent_text.place(segment.text, start, end)
            word_times.extend(placed)
        if final and sent_text is not None:
            # A sent word never splits across chunks, so its pieces end here.
            unfinished = sent_text.finish_word()
            word_times.extend(unfinished)
        if word_times:
            await self.add_word_timestamps(word_times, context_id=chunk.context_id)

    async def _alignment_settle_handler(self):
        """Report the latest chunk's last word once its alignment has gone quiet."""
        while True:
            await self._alignment_updated.wait()
            self._alignment_updated.clear()
            while True:
                remaining = self._last_alignment_time + ALIGNMENT_SETTLE_SECS - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(remaining)
            latest = self._latest_chunk
            if latest is None or latest not in self._chunks:
                continue
            try:
                await self._report_chunk(self._chunks[latest], final=True)
            except Exception as e:
                logger.warning(f"{self}: could not report Fish Audio word timestamps: {e}")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            audio_data = msg.get("audio")
                            context_id = self.get_active_audio_context_id()
                            if audio_data:
                                frame = TTSAudioRawFrame(
                                    audio_data,
                                    self.sample_rate,
                                    1,
                                    context_id=context_id,
                                )
                                await self.append_to_audio_context(context_id, frame)
                                await self.stop_ttfb_metrics()
                            if self._word_timestamps:
                                await self._handle_alignment(msg, context_id)
                        elif event == "finish":
                            await self._finish_all_chunks()
                            reason = msg.get("reason", "unknown")
                            if reason == "error":
                                await self.push_error(
                                    error_msg="Fish Audio server error during synthesis"
                                )
                            else:
                                logger.debug(f"Fish Audio session finished: {reason}")

            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        # The socket closed: report the words of audio already received.
        await self._finish_all_chunks()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Fish Audio's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames and control frames for the synthesized speech.
        """
        logger.debug(f"{self}: Generating Fish TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if self._word_timestamps:
                sent_text = self._context_text.get(context_id)
                if sent_text is None:
                    sent_text = _SentText()
                    self._context_text[context_id] = sent_text
                sent_text.add(text)

            # Send the text
            text_message = {
                "event": "text",
                "text": text,
            }
            try:
                await self._get_websocket().send(ormsgpack.packb(text_message))
                await self.start_tts_usage_metrics(text)

                # Send flush event to force audio generation
                flush_message = {"event": "flush"}
                await self._get_websocket().send(ormsgpack.packb(flush_message))
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


@dataclass
class _Segment:
    """One word of Fish Audio's alignment, timed from the start of its chunk."""

    text: str
    start: float
    end: float


@dataclass
class _ChunkAlignment:
    """A text chunk's latest alignment snapshot, and how many of its words were reported."""

    context_id: str
    offset: float
    segments: list[_Segment] = field(default_factory=list)
    reported: int = 0
    closed: bool = False


def _valid_segments(raw: Any) -> list[_Segment]:
    """The timed words of an alignment snapshot, skipping malformed entries."""
    segments = []
    if not isinstance(raw, list):
        return segments
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        start = item.get("start")
        end = item.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        segments.append(_Segment(text=str(text), start=float(start), end=float(end)))
    return segments


def _word_key(text: str) -> str:
    """A word's letters and digits, lowercased, for matching reported words to sent ones."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


@dataclass
class _SentToken:
    """One bracketed tag or one word of the text sent for an audio context.

    Parameters:
        text: The tag, or the word as it was sent.
        key: A word's letters and digits, lowercased; empty for a tag.
    """

    text: str
    key: str = ""

    @property
    def is_tag(self) -> bool:
        return not self.key


class _SentText:
    """The words and bracketed tags sent for one audio context, in order.

    Fish Audio reports the words it speaks but not the bracketed tags it
    performs, and it reports some sent words in pieces ("$42.50" as "42" and
    "50"). Word tracking expects every tag before a word to be reported as a
    word of its own, and pieces reported apart would reach the conversation
    context with spaces inside the word. So each word Fish reports is matched
    against the text sent: the tags passed over on the way are reported before
    it, and the pieces of one sent word are reported once, as the word sent.
    """

    # How many sent words a reported word may skip to find its match.
    LOOKAHEAD_WORDS = 4

    def __init__(self):
        self._tokens: list[_SentToken] = []
        self._position = 0
        # The pieces Fish has reported of the sent word at _position.
        self._pieces_key = ""
        self._pieces_start = 0.0
        self._pieces_end = 0.0
        # A sent word reported before Fish spoke all of it, and the pieces spoken so far.
        self._early_word: _SentToken | None = None
        self._early_key = ""
        self._last_end = 0.0

    def add(self, text: str):
        for match in _TEXT_TOKEN_RE.finditer(text):
            token = match.group(0)
            if token.startswith("["):
                self._tokens.append(_SentToken(text=token))
                continue
            if token.startswith("<"):
                continue
            key = _word_key(token)
            if key:
                self._tokens.append(_SentToken(text=token, key=key))

    def place(self, word: str, start: float, end: float) -> list[tuple[str, float, float]]:
        """What to report for a word Fish spoke: the tags sent before it, then the word.

        Each tag is timed at the end of the word before it. Nothing is returned
        for a piece of a sent word until Fish has spoken all of it.
        """
        reported = []
        key = _word_key(word)
        if self._continues_early_word(key):
            self._last_end = end
            return reported
        if self._pieces_key:
            sent_word = self._tokens[self._position]
            joined = self._pieces_key + key
            if key and sent_word.key.startswith(joined):
                self._pieces_key = joined
                self._pieces_end = end
                if joined == sent_word.key:
                    finished = self._finish_pieces()
                    reported.append(finished)
                self._last_end = end
                return reported
            # Fish moved on before speaking all of the sent word.
            finished = self._finish_pieces()
            reported.append(finished)
        if not key:
            reported.append((word, start, end))
            self._last_end = end
            return reported
        index, tags = self._find(key)
        if index is None:
            # No sent word matches: report the word alone and stay in place.
            reported.append((word, start, end))
            self._last_end = end
            return reported
        tag_time = min(self._last_end, start)
        for tag in tags:
            reported.append((tag, tag_time, tag_time))
        sent_word = self._tokens[index]
        if sent_word.key == key:
            self._position = index + 1
            reported.append((word, start, end))
        else:
            self._position = index
            self._pieces_key = key
            self._pieces_start = start
            self._pieces_end = end
        self._last_end = end
        return reported

    def finish_word(self) -> list[tuple[str, float, float]]:
        """Report the sent word Fish has spoken only part of, if any, as that word.

        Pieces of it Fish reports afterwards are taken as part of it, not reported again.
        """
        if not self._pieces_key:
            return []
        self._early_word = self._tokens[self._position]
        self._early_key = self._pieces_key
        finished = self._finish_pieces()
        return [finished]

    def _continues_early_word(self, key: str) -> bool:
        """Whether key is more of the sent word finish_word reported early."""
        if self._early_word is None:
            return False
        joined = self._early_key + key
        if not key or not self._early_word.key.startswith(joined):
            self._early_word = None
            return False
        self._early_key = joined
        if joined == self._early_word.key:
            self._early_word = None
        return True

    def _find(self, key: str) -> tuple[int | None, list[str]]:
        """The next sent word that starts with key, within the lookahead, and the tags before it."""
        tags = []
        words_seen = 0
        index = self._position
        while index < len(self._tokens) and words_seen < self.LOOKAHEAD_WORDS:
            token = self._tokens[index]
            if token.is_tag:
                tags.append(token.text)
                index += 1
                continue
            words_seen += 1
            if token.key.startswith(key):
                return index, tags
            index += 1
        return None, []

    def _finish_pieces(self) -> tuple[str, float, float]:
        sent_word = self._tokens[self._position]
        self._position += 1
        self._pieces_key = ""
        # Fish reports words without their trailing punctuation.
        spelling = sent_word.text.rstrip(".,!?;:")
        return (spelling, self._pieces_start, self._pieces_end)
