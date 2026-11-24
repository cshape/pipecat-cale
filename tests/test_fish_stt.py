#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio realtime STT against a local fake of the realtime ASR server.

The service runs in a test pipeline fed with real-time 20 ms audio frames. The
fake server answers ``config`` with ``session_started`` (or an error) and plays
a script of turn events keyed by how many audio messages it has received.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import ormsgpack
import pytest
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EagerTranscriptionFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextAssistantTurnFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SpeechControlParamsFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    UserBackchannelFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.services.fish.stt import (
    FishAudioSTTService,
    _sanitize_terms,
    _split_speakers,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    ExternalUserTurnStrategies,
)
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601

FRAME_SECONDS = 0.02


def _user_turn(index: int, text: str, *, start: int, taken: int, end: int, commit: int) -> dict:
    """Script one user turn: audio message counts for each event, and its words."""
    words = [{"text": word, "start_ms": start * 20, "end_ms": end * 20} for word in text.split()]
    return {
        start: [{"type": "speech_start", "turn_index": index, "audio_ms": start * 20}],
        start + 2: [{"type": "draft", "turn_index": index, "text": text.split()[0]}],
        taken: [{"type": "user_turn_taken", "turn_index": index, "audio_ms": taken * 20}],
        commit: [
            {
                "type": "committed",
                "turn_index": index,
                "text": text,
                "words": words,
                "language_code": "en",
                "audio_start_ms": start * 20,
                "audio_end_ms": commit * 20,
                "reason": "eot",
                "user_turn": True,
                "turn_kind": "user",
            }
        ],
    }


@dataclass
class _FakeFishASR:
    # Events to send per connection, keyed by the count of audio messages received.
    scripts: list[dict[int, list[dict]]] = field(default_factory=list)
    # Close each connection with this code after this many audio messages.
    close_after: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Refuse a config that has results.text_format, as transcribe-stream-1 does.
    refuse_text_format: bool = False
    # Refuse every config.
    refuse_config: bool = False
    # Answer every configure with ok: false.
    refuse_configure: bool = False
    # Never send session_started.
    never_start: bool = False
    served_model: str | None = None
    # The turn detection mode the server reports, whatever the config asked for.
    served_mode: str | None = None
    headers: list[dict] = field(default_factory=list)
    configs: list[dict] = field(default_factory=list)
    messages: list[list[dict]] = field(default_factory=list)
    audio_counts: list[int] = field(default_factory=list)
    audio_bytes: list[int] = field(default_factory=list)

    async def handle(self, websocket):
        index = len(self.configs)
        self.headers.append(dict(websocket.request.headers))
        self.messages.append([])
        self.audio_counts.append(0)
        self.audio_bytes.append(0)
        config = ormsgpack.unpackb(await websocket.recv())
        self.configs.append(config)
        if self.refuse_config or (
            self.refuse_text_format and "text_format" in config.get("results", {})
        ):
            await self._send(
                websocket,
                {
                    "type": "error",
                    "code": 400,
                    "error_type": "invalid_config",
                    "message": "unknown field config.results.text_format",
                    "recoverable": False,
                },
            )
            await websocket.close(4400)
            return
        model = self.served_model or websocket.request.headers.get("model")
        mode = self.served_mode or config.get("turn_detection", {}).get("mode", "auto")
        if not self.never_start:
            await self._send(
                websocket,
                {
                    "type": "session_started",
                    "session_id": f"s_{index}",
                    "request_id": f"r_{index}",
                    "config": {"model": model, "turn_detection": {"mode": mode}},
                    "limits": {},
                },
            )
        script = self.scripts[index] if index < len(self.scripts) else {}
        open_turn = None
        async for raw in websocket:
            message = ormsgpack.unpackb(raw)
            kind = message.get("type")
            if kind != "audio":
                self.messages[index].append(message)
            if kind == "configure":
                if self.refuse_configure:
                    reply = {"type": "configured", "ok": False, "message": "refused"}
                else:
                    turn_detection = message.get("turn_detection") or {}
                    mode = turn_detection.get("mode", mode)
                    reply = {
                        "type": "configured",
                        "ok": True,
                        "config": {"turn_detection": {"mode": mode}},
                    }
                await self._send(websocket, reply)
            if kind == "audio":
                self.audio_counts[index] += 1
                self.audio_bytes[index] += len(message["data"])
                count = self.audio_counts[index]
                for event in script.get(count, []):
                    if event["type"] == "speech_start":
                        open_turn = event["turn_index"]
                    if event["type"] == "committed":
                        open_turn = None
                    await self._send(websocket, event)
                close = self.close_after.get(index)
                if close is not None and count == close[0]:
                    await websocket.close(close[1])
                    return
            elif kind == "finalize" and open_turn is not None:
                await self._commit(websocket, open_turn, "finalize")
                open_turn = None
            elif kind == "audio_done":
                if open_turn is not None:
                    await self._commit(websocket, open_turn, "audio_done")
                await self._send(websocket, {"type": "session_done", "stats": {}})
                await websocket.close(1000)
                return

    async def _commit(self, websocket, turn_index: int, reason: str):
        await self._send(
            websocket,
            {
                "type": "committed",
                "turn_index": turn_index,
                "text": "held text",
                "words": [{"text": "held", "start_ms": 0, "end_ms": 100}],
                "audio_start_ms": 0,
                "audio_end_ms": 200,
                "reason": reason,
                "user_turn": True,
                "turn_kind": "user",
            },
        )

    @staticmethod
    async def _send(websocket, message: dict[str, Any]):
        await websocket.send(ormsgpack.packb(message))


def _audio_frames(seconds: float, after: list | None = None) -> list:
    frames = []
    for _ in range(int(seconds / FRAME_SECONDS)):
        frames.append(
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
        )
        frames.append(SleepFrame(sleep=FRAME_SECONDS))
    if after:
        frames.extend(after)
    return frames


async def _run(server: _FakeFishASR, frames: list, observers=None, **service_kwargs):
    async with serve(server.handle, "127.0.0.1", 0) as s:
        port = s.sockets[0].getsockname()[1]
        service = FishAudioSTTService(
            api_key="test-key", url=f"ws://127.0.0.1:{port}", **service_kwargs
        )
        service._reconnect_backoff_min_wait = 0.01
        service._reconnect_backoff_max_wait = 0.01
        down, up = await run_test(service, frames_to_send=frames, observers=observers or [])
        await asyncio.sleep(0.1)
    return service, down, up


def _service(**kwargs) -> FishAudioSTTService:
    service = FishAudioSTTService(api_key="test-key", **kwargs)
    service._sample_rate = 16000
    return service


# ----------------------------------------------------------------------
# Settings and config
# ----------------------------------------------------------------------


def test_config_defaults():
    config = _service()._session_config()

    assert config["audio"] == {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    assert config["turn_detection"] == {"mode": "auto"}
    assert config["results"] == {
        "drafts": True,
        "ignore_timestamps": False,
        "diarize": False,
        "tag_audio_events": False,
    }
    assert "context" not in config


def test_config_carries_eager_text_format_terms_and_silence():
    service = _service(
        enable_eager_end_of_turn=True,
        settings=FishAudioSTTService.Settings(
            model="transcribe-stream-2",
            text_format="written",
            keyterms=["Fish Audio", "Pipecat"],
            silence_timeout_ms=4000,
        ),
    )

    config = service._session_config()

    assert config["turn_detection"] == {"mode": "auto", "eager": True, "silence_timeout_ms": 4000}
    assert config["results"]["text_format"] == "written"
    assert config["context"] == {"terms": ["Fish Audio", "Pipecat"]}


def test_send_sample_rate_falls_back_to_16k_for_rates_fish_does_not_take():
    service = _service()
    service._sample_rate = 11025

    assert service.send_sample_rate == 16000


def test_keyterms_are_sanitized_to_fish_limits():
    terms = ["ok", " ok ", "x" * 33, ""] + [f"t{i}" for i in range(120)]

    sanitized = _sanitize_terms(terms)

    assert sanitized[0] == "ok"
    assert "x" * 33 not in sanitized
    assert len(sanitized) == 100


def test_silence_timeout_outside_the_mode_range_is_refused():
    with pytest.raises(ValueError, match="between 3000 and 60000"):
        _service(settings=FishAudioSTTService.Settings(silence_timeout_ms=700))

    service = _service(
        settings=FishAudioSTTService.Settings(turn_detection_mode="vad", silence_timeout_ms=700)
    )
    assert service._session_config()["turn_detection"] == {"mode": "vad", "silence_timeout_ms": 700}


def test_eager_and_vad_force_turn_endpoint_do_not_go_together():
    with pytest.raises(ValueError):
        _service(enable_eager_end_of_turn=True, vad_force_turn_endpoint=True)


def test_recommended_turn_strategies():
    default = _service().service_metadata_frame().user_turn_strategies
    eager = _service(enable_eager_end_of_turn=True).service_metadata_frame().user_turn_strategies
    forced = _service(vad_force_turn_endpoint=True).service_metadata_frame().user_turn_strategies

    assert isinstance(default, ExternalUserTurnStrategies)
    assert not isinstance(default, EagerUserTurnStrategies)
    assert isinstance(eager, EagerUserTurnStrategies)
    assert forced is None


def test_split_speakers():
    runs = _split_speakers("<|speaker:0|>Hi there. <|speaker:1|> Hello!")

    assert runs == [("0", "Hi there."), ("1", "Hello!")]


@pytest.mark.asyncio
async def test_updates_go_to_the_open_session_or_reconnect(monkeypatch):
    service = _service()
    service._session_confirmed.set()
    service._websocket = AsyncMock()
    service._websocket.state = State.OPEN
    sent = []

    async def capture(message):
        sent.append(message)

    monkeypatch.setattr(service, "_send", capture)
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(
        FishAudioSTTService.Settings(turn_detection_mode="vad", silence_timeout_ms=2000)
    )
    await service._update_settings(FishAudioSTTService.Settings(keyterms=["Lisbon"]))
    await service._update_settings(FishAudioSTTService.Settings(model="transcribe-stream-2"))

    assert sent == [
        {"type": "configure", "turn_detection": {"mode": "vad", "silence_timeout_ms": 2000}},
        {"type": "configure", "context": {"terms": ["Lisbon"]}},
    ]
    reconnect.assert_awaited_once()


# ----------------------------------------------------------------------
# Turn handling
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_user_turn_becomes_proposed_turn_frames_and_a_final_transcript():
    server = _FakeFishASR(
        scripts=[_user_turn(0, "Book a table", start=3, taken=6, end=10, commit=20)]
    )

    _, down, _ = await _run(server, _audio_frames(0.6))

    kinds = [
        type(f)
        for f in down
        if isinstance(
            f,
            (
                ProposedUserStartedSpeakingFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
                ProposedUserStoppedSpeakingFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
            ),
        )
    ]
    assert kinds == [
        VADUserStartedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
        InterimTranscriptionFrame,
        VADUserStoppedSpeakingFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    final = next(f for f in down if isinstance(f, TranscriptionFrame))
    assert final.text == "Book a table"
    assert final.finalized
    assert final.language.value == "en"
    assert server.headers[0]["model"] == "transcribe-stream-1"


@pytest.mark.asyncio
async def test_vad_frames_place_the_real_start_and_end_of_speech():
    server = _FakeFishASR(
        scripts=[_user_turn(0, "Book a table", start=3, taken=25, end=10, commit=30)]
    )

    _, down, _ = await _run(server, _audio_frames(0.8))

    started = next(f for f in down if isinstance(f, VADUserStartedSpeakingFrame))
    stopped = next(f for f in down if isinstance(f, VADUserStoppedSpeakingFrame))
    # Speech started at message 3 and was taken at message 25: 22 messages back.
    assert 0.3 < started.start_secs < 0.6
    # The last word ended at message 10 and the commit came at message 30.
    assert 0.3 < stopped.stop_secs < 0.6


@pytest.mark.asyncio
async def test_latency_is_measured_without_a_local_vad():
    server = _FakeFishASR(
        scripts=[_user_turn(0, "Book a table", start=3, taken=6, end=10, commit=20)]
    )
    observer = UserBotLatencyObserver()
    latencies = []

    @observer.event_handler("on_latency_measured")
    async def on_latency_measured(_observer, latency):
        latencies.append(latency)

    frames = _audio_frames(0.5, after=[SleepFrame(sleep=0.1), BotStartedSpeakingFrame()])
    await _run(server, frames, observers=[observer])

    assert len(latencies) == 1
    # Speech ended at message 10; the bot spoke ~0.5 s of audio and 0.1 s later.
    assert 0.4 < latencies[0] < 1.0


@pytest.mark.asyncio
async def test_a_pipeline_vad_stops_the_vad_frames():
    server = _FakeFishASR(
        scripts=[_user_turn(0, "Book a table", start=5, taken=8, end=10, commit=20)]
    )

    frames = [SpeechControlParamsFrame(vad_params=VADParams())] + _audio_frames(0.6)
    _, down, _ = await _run(server, frames)

    assert not any(
        isinstance(f, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)) for f in down
    )
    assert any(isinstance(f, TranscriptionFrame) for f in down)


@pytest.mark.asyncio
async def test_turns_that_are_not_the_users_stay_out():
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        10: [
            {
                "type": "committed",
                "turn_index": 0,
                "text": "[tv noise]",
                "user_turn": False,
                "turn_kind": "other",
                "reason": "eot",
            }
        ],
        12: [{"type": "speech_start", "turn_index": 1, "audio_ms": 240}],
        16: [
            {
                "type": "committed",
                "turn_index": 1,
                "text": "Mm-hm.",
                "user_turn": False,
                "turn_kind": "backchannel",
                "reason": "eot",
            }
        ],
    }
    server = _FakeFishASR(scripts=[script])

    _, down, _ = await _run(server, _audio_frames(0.5))

    assert not any(
        isinstance(f, (TranscriptionFrame, ProposedUserStartedSpeakingFrame)) for f in down
    )
    backchannels = [f.text for f in down if isinstance(f, UserBackchannelFrame)]
    assert backchannels == ["Mm-hm."]


@pytest.mark.asyncio
async def test_eager_end_of_turn_and_its_withdrawal():
    script = _user_turn(0, "Book a table", start=3, taken=5, end=10, commit=24)
    script[12] = [{"type": "eager_end_of_turn", "turn_index": 0, "text": "Book a"}]
    script[14] = [{"type": "turn_resumed", "turn_index": 0}]
    script[18] = [{"type": "eager_end_of_turn", "turn_index": 0, "text": "Book a table"}]
    server = _FakeFishASR(scripts=[script])

    _, down, _ = await _run(server, _audio_frames(0.6), enable_eager_end_of_turn=True)

    eager = [f.text for f in down if isinstance(f, EagerTranscriptionFrame)]
    assert eager == ["Book a", "Book a table"]
    assert server.configs[0]["turn_detection"]["eager"] is True


@pytest.mark.asyncio
async def test_vad_force_turn_endpoint_finalizes_on_each_vad_stop():
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        5: [{"type": "user_turn_taken", "turn_index": 0, "audio_ms": 100}],
    }
    server = _FakeFishASR(scripts=[script])
    frames = _audio_frames(0.2, after=[VADUserStoppedSpeakingFrame(stop_secs=0.2)])
    frames.extend(_audio_frames(0.2))

    _, down, _ = await _run(server, frames, vad_force_turn_endpoint=True)

    assert {"type": "finalize"} in server.messages[0]
    finals = [(f.text, f.finalized) for f in down if isinstance(f, TranscriptionFrame)]
    assert finals == [("held text", True)]
    assert not any(
        isinstance(f, (ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame))
        for f in down
    )


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stopping_lets_the_server_commit_the_open_turn():
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        5: [{"type": "user_turn_taken", "turn_index": 0, "audio_ms": 100}],
    }
    server = _FakeFishASR(scripts=[script])

    _, down, _ = await _run(server, _audio_frames(0.3))

    assert {"type": "audio_done"} in server.messages[0]
    assert [f.text for f in down if isinstance(f, TranscriptionFrame)] == ["held text"]


@pytest.mark.asyncio
async def test_a_restarted_session_is_replaced_and_seeded_with_the_conversation():
    first = _user_turn(0, "Book a table", start=3, taken=5, end=8, commit=10)
    server = _FakeFishASR(scripts=[first, {}], close_after={0: (20, 1012)})
    frames = _audio_frames(
        0.3,
        after=[LLMContextAssistantTurnFrame(text="For how many?", timestamp=time_now_iso8601())],
    )
    frames.extend(_audio_frames(0.4))

    service, _, _ = await _run(server, frames)

    assert len(server.configs) == 2
    assert server.configs[1]["context"]["history"] == [
        {"role": "user", "text": "Book a table"},
        {"role": "assistant", "text": "For how many?"},
    ]
    assert {"type": "context_add", "items": [{"role": "assistant", "text": "For how many?"}]} in (
        server.messages[0]
    )
    assert service.is_usable


@pytest.mark.asyncio
async def test_text_format_refused_by_the_model_reconnects_without_it():
    server = _FakeFishASR(refuse_text_format=True)
    settings = FishAudioSTTService.Settings(model="transcribe-stream-2", text_format="written")

    service, _, up = await _run(server, _audio_frames(0.4), settings=settings)

    assert not any(isinstance(f, ErrorFrame) for f in up)
    assert len(server.configs) == 2
    assert "text_format" in server.configs[0]["results"]
    assert "text_format" not in server.configs[1]["results"]
    assert server.audio_counts[1] > 0
    assert service.is_usable


@pytest.mark.asyncio
async def test_a_refused_config_is_not_retried():
    server = _FakeFishASR(refuse_config=True)

    service, _, up = await _run(server, _audio_frames(0.3))

    assert len(server.configs) == 1
    assert not service.is_usable
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors[0].category == ErrorCategory.INVALID_REQUEST
    assert "invalid_config" in errors[0].error


@pytest.mark.asyncio
async def test_a_fallback_model_is_logged(caplog):
    server = _FakeFishASR(served_model="transcribe-stream-1")
    settings = FishAudioSTTService.Settings(model="transcribe-stream-2")
    sessions = []

    def capture(service):
        @service.event_handler("on_session_started")
        async def on_session_started(_service, data):
            sessions.append(data["config"]["model"])

    async with serve(server.handle, "127.0.0.1", 0) as s:
        port = s.sockets[0].getsockname()[1]
        service = FishAudioSTTService(
            api_key="test-key", url=f"ws://127.0.0.1:{port}", settings=settings
        )
        capture(service)
        await run_test(service, frames_to_send=_audio_frames(0.2))

    assert server.headers[0]["model"] == "transcribe-stream-2"
    assert sessions == ["transcribe-stream-1"]


# ----------------------------------------------------------------------
# Runtime settings
# ----------------------------------------------------------------------


def _capturing_service(monkeypatch, **kwargs) -> tuple[FishAudioSTTService, list]:
    """A service with a confirmed session whose sent messages are captured."""
    service = _service(**kwargs)
    service._websocket = AsyncMock()
    service._websocket.state = State.OPEN
    service._session_confirmed.set()
    sent = []

    async def capture(message):
        sent.append(message)

    monkeypatch.setattr(service, "_send", capture)
    return service, sent


@pytest.mark.asyncio
async def test_a_mode_change_without_a_timeout_returns_to_the_modes_default(monkeypatch):
    settings = FishAudioSTTService.Settings(turn_detection_mode="vad", silence_timeout_ms=700)
    service, sent = _capturing_service(monkeypatch, settings=settings)

    await service._update_settings(FishAudioSTTService.Settings(turn_detection_mode="auto"))

    assert sent == [
        {"type": "configure", "turn_detection": {"mode": "auto", "silence_timeout_ms": 5000}}
    ]
    assert service._settings.silence_timeout_ms is None
    assert service._session_config()["turn_detection"] == {"mode": "auto"}


@pytest.mark.asyncio
async def test_an_invalid_mode_and_timeout_pair_is_refused(monkeypatch):
    service, sent = _capturing_service(monkeypatch)

    with pytest.raises(ValueError, match="between 3000 and 60000"):
        await service._update_settings(FishAudioSTTService.Settings(silence_timeout_ms=700))
    with pytest.raises(ValueError, match="turn_detection_mode"):
        await service._update_settings(FishAudioSTTService.Settings(turn_detection_mode=None))

    assert sent == []
    assert service._settings.turn_detection_mode == "auto"


def test_a_missing_turn_detection_mode_is_refused_at_construction():
    with pytest.raises(ValueError, match="turn_detection_mode"):
        _service(settings=FishAudioSTTService.Settings(turn_detection_mode=None))


@pytest.mark.asyncio
async def test_clearing_the_timeout_sends_the_modes_default(monkeypatch):
    settings = FishAudioSTTService.Settings(turn_detection_mode="vad", silence_timeout_ms=900)
    service, sent = _capturing_service(monkeypatch, settings=settings)

    await service._update_settings(FishAudioSTTService.Settings(silence_timeout_ms=None))

    assert sent == [
        {"type": "configure", "turn_detection": {"mode": "vad", "silence_timeout_ms": 700}}
    ]


@pytest.mark.asyncio
async def test_switching_to_auto_keeps_eager(monkeypatch):
    settings = FishAudioSTTService.Settings(turn_detection_mode="vad")
    service, sent = _capturing_service(
        monkeypatch, enable_eager_end_of_turn=True, settings=settings
    )

    await service._update_settings(FishAudioSTTService.Settings(turn_detection_mode="auto"))

    assert sent[0]["turn_detection"] == {"mode": "auto", "eager": True, "silence_timeout_ms": 5000}


@pytest.mark.asyncio
async def test_a_model_change_tries_text_format_again(monkeypatch):
    settings = FishAudioSTTService.Settings(text_format="written")
    service = _service(settings=settings)
    service._text_format_refused = True
    monkeypatch.setattr(service, "_request_reconnect", AsyncMock())

    await service._update_settings(FishAudioSTTService.Settings(model="transcribe-stream-2"))

    assert service._session_config()["results"]["text_format"] == "written"


@pytest.mark.asyncio
async def test_settings_changed_before_the_session_starts_reach_it(monkeypatch):
    service, sent = _capturing_service(monkeypatch)
    service._session_confirmed.clear()

    await service._update_settings(FishAudioSTTService.Settings(keyterms=["Lisbon"]))
    assert sent == []

    await service._handle_session_started({"session_id": "s", "config": {}})

    assert sent == [{"type": "configure", "context": {"terms": ["Lisbon"]}}]


@pytest.mark.asyncio
async def test_a_refused_configure_rolls_the_settings_back():
    server = _FakeFishASR(refuse_configure=True)
    settings = FishAudioSTTService.Settings(turn_detection_mode="vad", silence_timeout_ms=700)
    update = FishAudioSTTService.Settings(turn_detection_mode="auto")
    frames = _audio_frames(0.2, after=[STTUpdateSettingsFrame(delta=update)])
    frames.extend(_audio_frames(0.2))

    service, _, _ = await _run(server, frames, settings=settings)

    assert {
        "type": "configure",
        "turn_detection": {"mode": "auto", "silence_timeout_ms": 5000},
    } in (server.messages[0])
    assert service._settings.turn_detection_mode == "vad"
    assert service._settings.silence_timeout_ms == 700


@pytest.mark.asyncio
async def test_a_model_change_waits_for_the_open_turn_and_loses_no_audio():
    first = _user_turn(0, "Book a table", start=3, taken=6, end=20, commit=30)
    server = _FakeFishASR(scripts=[first, {}])
    update = FishAudioSTTService.Settings(model="transcribe-stream-2")
    frames = _audio_frames(0.2, after=[STTUpdateSettingsFrame(delta=update)])
    frames.extend(_audio_frames(1.0))
    sent_bytes = sum(len(f.audio) for f in frames if isinstance(f, InputAudioRawFrame))

    _, down, _ = await _run(server, frames)

    # The turn committed on the first session; the reconnect came after it.
    assert server.audio_counts[0] >= 30
    assert [h["model"] for h in server.headers] == ["transcribe-stream-1", "transcribe-stream-2"]
    finals = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert finals == ["Book a table"]
    assert sum(server.audio_bytes) >= sent_bytes * 0.97


# ----------------------------------------------------------------------
# Turns without words, and turns the server decides otherwise
# ----------------------------------------------------------------------


def _turn_frames(down) -> list[type]:
    kinds = (
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
        TranscriptionFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    return [type(f) for f in down if isinstance(f, kinds)]


@pytest.mark.asyncio
@pytest.mark.parametrize("tag_audio_events, text", [(False, ""), (True, "[laughter]")])
async def test_a_turn_without_words_neither_interrupts_nor_opens_a_turn(tag_audio_events, text):
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        5: [{"type": "user_turn_taken", "turn_index": 0, "audio_ms": 100}],
        7: [{"type": "draft", "turn_index": 0, "text": text}],
        12: [
            {
                "type": "committed",
                "turn_index": 0,
                "text": text,
                "words": [],
                "reason": "eot",
                "user_turn": True,
                "turn_kind": "user",
            }
        ],
    }
    server = _FakeFishASR(scripts=[script])
    settings = FishAudioSTTService.Settings(tag_audio_events=tag_audio_events)

    _, down, _ = await _run(server, _audio_frames(0.4), settings=settings)

    assert _turn_frames(down) == []


@pytest.mark.asyncio
async def test_a_turn_starts_on_its_first_draft_with_words():
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        5: [{"type": "user_turn_taken", "turn_index": 0, "audio_ms": 100}],
        8: [{"type": "draft", "turn_index": 0, "text": "Book"}],
    }
    server = _FakeFishASR(scripts=[script])
    service = None

    async def watch():
        await asyncio.sleep(0.13)
        return service._user_turn_open

    async with serve(server.handle, "127.0.0.1", 0) as s:
        port = s.sockets[0].getsockname()[1]
        service = FishAudioSTTService(api_key="test-key", url=f"ws://127.0.0.1:{port}")
        open_after_taken = asyncio.create_task(watch())
        down, _ = await run_test(service, frames_to_send=_audio_frames(0.3))

    assert not await open_after_taken
    kinds = [
        type(f)
        for f in down
        if isinstance(f, (ProposedUserStartedSpeakingFrame, InterimTranscriptionFrame))
    ]
    assert kinds == [ProposedUserStartedSpeakingFrame, InterimTranscriptionFrame]


@pytest.mark.asyncio
async def test_the_servers_mode_decides_and_a_non_user_commit_closes_an_announced_turn():
    # The config asks for auto mode, but the server runs the session in vad mode.
    script = {
        3: [{"type": "speech_start", "turn_index": 0, "audio_ms": 60}],
        5: [{"type": "draft", "turn_index": 0, "text": "Book a"}],
        12: [
            {
                "type": "committed",
                "turn_index": 0,
                "text": "Book a",
                "words": [{"text": "Book", "start_ms": 60, "end_ms": 160}],
                "reason": "eot",
                "user_turn": False,
                "turn_kind": "other",
            }
        ],
    }
    server = _FakeFishASR(scripts=[script], served_mode="vad")

    service, down, _ = await _run(server, _audio_frames(0.4))

    assert _turn_frames(down) == [
        VADUserStartedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert not service._user_turn_open


@pytest.mark.asyncio
async def test_a_commit_without_word_timings_still_ends_the_vad_speech():
    script = _user_turn(0, "Book a table", start=3, taken=6, end=10, commit=20)
    script[20][0]["words"] = []
    server = _FakeFishASR(scripts=[script])

    _, down, _ = await _run(server, _audio_frames(0.6))

    assert _turn_frames(down) == [
        VADUserStartedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]


# ----------------------------------------------------------------------
# Finalize and stop
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_without_an_open_turn_waits_for_no_final(monkeypatch):
    service, sent = _capturing_service(monkeypatch)

    await service.force_end_turn()

    assert sent == [{"type": "finalize"}]
    assert not service._finalize_requested


@pytest.mark.asyncio
async def test_stopping_without_a_session_does_not_wait_for_one():
    server = _FakeFishASR(never_start=True)

    started = time.monotonic()
    await _run(server, _audio_frames(0.2))

    assert time.monotonic() - started < 1.5
    assert {"type": "audio_done"} not in server.messages[0]
