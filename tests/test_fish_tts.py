#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Fish Audio TTS."""

import asyncio
from unittest.mock import AsyncMock

import ormsgpack
import pytest

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.adapters.tts.services.fish_adapter import (
    FishAudioDrama3ExpressionAdapter,
    FishAudioS2ExpressionAdapter,
)
from pipecat.services.fish import tts as fish_tts
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.utils.types import NOT_GIVEN


@pytest.mark.asyncio
async def test_one_silent_context_writes_off_the_service():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=1)

    assert service._max_consecutive_zero_audio_contexts == 1


@pytest.mark.asyncio
async def test_the_silent_context_limit_can_be_raised():
    service = FishAudioTTSService(api_key="key", max_consecutive_zero_audio_contexts=4)

    assert service._max_consecutive_zero_audio_contexts == 4


async def _start_request(settings=None):
    """Connect with a mocked socket and return the start message's request."""
    service = FishAudioTTSService(api_key="key", settings=settings)
    websocket = AsyncMock()
    service._websocket_connect = AsyncMock(return_value=websocket)

    await service._connect_websocket()

    message = ormsgpack.unpackb(websocket.send.await_args.args[0])
    assert message["event"] == "start"
    return message["request"]


async def _connect_headers(settings=None):
    """Connect with a mocked socket and return the handshake headers."""
    service = FishAudioTTSService(api_key="key", settings=settings)
    service._websocket_connect = AsyncMock(return_value=AsyncMock())

    await service._connect_websocket()

    return service._websocket_connect.await_args.kwargs["additional_headers"]


@pytest.mark.asyncio
async def test_default_model_is_s2_1_pro():
    headers = await _connect_headers()

    assert headers["model"] == "s2.1-pro"


@pytest.mark.asyncio
async def test_drama_3_model_is_sent_in_handshake():
    headers = await _connect_headers(FishAudioTTSService.Settings(model="drama-3"))

    assert headers["model"] == "drama-3"


@pytest.mark.asyncio
async def test_socket_accepts_audio_messages_over_1_mib():
    service = FishAudioTTSService(api_key="key")
    service._websocket_connect = AsyncMock(return_value=AsyncMock())

    await service._connect_websocket()

    assert service._websocket_connect.await_args.kwargs["max_size"] == 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_start_message_omits_unset_request_settings():
    request = await _start_request()

    for name in ("chunk_length", "min_chunk_length", "condition_on_previous_chunks"):
        assert name not in request
    assert request["prosody"] == {"speed": 1.0, "volume": 0}


@pytest.mark.asyncio
async def test_start_message_carries_set_request_settings():
    request = await _start_request(
        FishAudioTTSService.Settings(
            chunk_length=150,
            min_chunk_length=20,
            condition_on_previous_chunks=False,
            prosody_normalize_loudness=False,
        )
    )

    assert request["chunk_length"] == 150
    assert request["min_chunk_length"] == 20
    assert request["condition_on_previous_chunks"] is False
    assert request["prosody"]["normalize_loudness"] is False
    assert "normalize_loudness" not in request


def test_nested_prosody_mapping_leaves_missing_keys_unset():
    delta = FishAudioTTSService.Settings.from_mapping({"prosody": {"speed": 1.2}})

    assert delta.prosody_speed == 1.2
    assert delta.prosody_volume is NOT_GIVEN
    assert delta.prosody_normalize_loudness is NOT_GIVEN


def test_nested_prosody_mapping_carries_normalize_loudness():
    delta = FishAudioTTSService.Settings.from_mapping({"prosody": {"normalize_loudness": False}})

    assert delta.prosody_normalize_loudness is False


class _Messages:
    """A websocket stand-in that yields the given messages, then ends."""

    def __init__(self, messages):
        self._messages = list(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


@pytest.mark.asyncio
async def test_short_audio_chunks_are_kept():
    # A decoded chunk is PCM, so a short one is real audio (an utterance's tail).
    service = FishAudioTTSService(api_key="key")
    chunks = [b"\x01\x00" * 160, b"\x02\x00" * 1024]
    service._websocket = _Messages(
        ormsgpack.packb({"event": "audio", "audio": chunk}) for chunk in chunks
    )
    service.get_active_audio_context_id = lambda: "ctx"
    service.stop_ttfb_metrics = AsyncMock()
    appended = []

    async def append_to_audio_context(context_id, frame):
        appended.append(frame.audio)

    service.append_to_audio_context = append_to_audio_context

    await service._receive_messages()

    assert appended == chunks


class TestS2ExpressionAdapter:
    def test_capabilities(self):
        capabilities = FishAudioS2ExpressionAdapter().capabilities()
        assert capabilities.allows_free_text_sounds
        assert capabilities.allows_free_text_delivery
        assert capabilities.emphasis
        assert "laughing" in [term.name for term in capabilities.sounds]
        # A delivery tag separates its names by spaces.
        assert all(len(term.name.split()) == 1 for term in capabilities.delivery)

    def test_slow_and_fast_are_written_as_speaking_tags(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert {"slow", "fast"} <= {term.name for term in adapter.capabilities().delivery}
        assert adapter.format_text("Hi.", ["calm", "slow"], None) == (
            " [calm] [speaking slowly] Hi."
        )
        assert adapter.format_text("Hi.", ["fast"], None) == " [speaking quickly] Hi."

    def test_a_pace_is_set_back_where_a_speed_delivery_ends(self):
        adapter = FishAudioS2ExpressionAdapter()
        # Within a request, the words after the slow ones set the pace back...
        assert adapter.format_text(", thanks.", ["calm"], ["calm", "slow"]) == (
            " [normal pace] [calm] , thanks."
        )
        # ...and a request that ends slow sets it back at its end.
        assert adapter.format_end(["calm", "slow"]) == " [normal pace]"
        assert adapter.format_end(["calm"]) == ""

    def test_tags_where_delivery_changes(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.format_text("Hi.", ["excited"], None) == " [excited] Hi."
        assert adapter.format_text("Hi.", ["excited"], ["excited"]) == "Hi."
        assert adapter.format_text("Hi.", ["sad", "whispering"], None) == (
            " [sad] [whispering] Hi."
        )

    def test_return_to_baseline_writes_no_tag(self):
        # Fish reads a [neutral] tag as a delivery of its own.
        assert FishAudioS2ExpressionAdapter().format_text("Hi.", [], ["excited"]) == "Hi."

    def test_emphasis(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.format_text("now", ["emphasis"], None) == " [emphasis] now"
        assert adapter.format_text("now", ["excited", "emphasis"], ["excited"]) == (
            " [emphasis] now"
        )

    def test_sounds_and_pauses(self):
        adapter = FishAudioS2ExpressionAdapter()
        assert adapter.format_sound("clear throat") == " [clear throat] "
        assert adapter.format_pause(600) == " [break] "
        assert adapter.format_pause(1200) == " [long-break] "


@pytest.mark.parametrize(
    "model, adapter",
    [
        ("s2.1-pro", FishAudioS2ExpressionAdapter),
        ("s2-pro", FishAudioS2ExpressionAdapter),
        ("s3-pro", FishAudioS2ExpressionAdapter),
        ("s1", BaseExpressionAdapter),
        ("drama-3", FishAudioDrama3ExpressionAdapter),
    ],
)
def test_expression_adapter_follows_the_model(model, adapter):
    service = FishAudioTTSService(api_key="key", settings=FishAudioTTSService.Settings(model=model))
    assert type(service.get_expression_adapter()) is adapter


class TestDrama3ExpressionAdapter:
    def test_capabilities(self):
        capabilities = FishAudioDrama3ExpressionAdapter().capabilities()
        assert capabilities.allows_free_text_sounds
        assert capabilities.allows_free_text_delivery
        assert capabilities.emphasis
        assert "sigh" in [term.name for term in capabilities.sounds]
        # A delivery tag separates its names by spaces.
        assert all(len(term.name.split()) == 1 for term in capabilities.delivery)

    def test_direction_where_delivery_changes(self):
        adapter = FishAudioDrama3ExpressionAdapter()
        assert adapter.format_text("Hi.", ["excited"], None) == " [excited] Hi."
        assert adapter.format_text("Hi.", ["excited"], ["excited"]) == "Hi."
        assert adapter.format_text("Hi.", ["sad", "nervous"], None) == " [sad, nervous] Hi."

    def test_return_to_baseline_writes_no_direction(self):
        assert FishAudioDrama3ExpressionAdapter().format_text("Hi.", [], ["excited"]) == "Hi."

    def test_word_level_delivery_is_a_paired_tag(self):
        adapter = FishAudioDrama3ExpressionAdapter()
        assert adapter.format_text("Go on.", ["whispering"], None) == (
            "<whisper> Go on. </whisper>"
        )
        assert adapter.format_text("so happy", ["excited", "slow"], ["excited"]) == (
            "<slow> so happy </slow>"
        )
        assert adapter.format_text("Go.", ["sad", "fast"], None) == " [sad] <fast> Go. </fast>"

    def test_a_paired_tag_needs_no_closing_request(self):
        assert FishAudioDrama3ExpressionAdapter().format_end(["slow"]) == ""

    def test_emphasis(self):
        adapter = FishAudioDrama3ExpressionAdapter()
        assert adapter.format_text("now", ["emphasis"], None) == "<emphasis> now </emphasis>"

    def test_sounds_and_pauses(self):
        adapter = FishAudioDrama3ExpressionAdapter()
        assert adapter.format_sound("shaky breath") == " [shaky breath] "
        assert adapter.format_pause(400) == " [short pause] "
        assert adapter.format_pause(1500) == " [long pause] "


def test_word_timestamps_use_the_timestamped_endpoint():
    service = FishAudioTTSService(api_key="key")

    assert service._base_url == "wss://api.fish.audio/v1/tts/live/with-timestamp"
    assert service._push_text_frames is False


def test_word_timestamps_can_be_turned_off():
    service = FishAudioTTSService(api_key="key", word_timestamps=False)

    assert service._base_url == "wss://api.fish.audio/v1/tts/live"
    assert service._push_text_frames is True


def _audio(chunk_seq, offset, words, audio=b"\x00\x00" * 240):
    segments = [{"text": text, "start": start, "end": end} for text, start, end in words]
    alignment = {"segments": segments, "audio_duration": 1.0}
    return ormsgpack.packb(
        {
            "event": "audio",
            "audio": audio,
            "content": "",
            "alignment": alignment,
            "chunk_seq": chunk_seq,
            "chunk_audio_offset_sec": offset,
        }
    )


class _Recorder:
    """A service whose reported word timestamps are recorded instead of played."""

    def __init__(self):
        self.service = FishAudioTTSService(api_key="key")
        self.service.stop_ttfb_metrics = AsyncMock()
        self.service.append_to_audio_context = AsyncMock()
        self.reported = []
        self.service.add_word_timestamps = self._add_word_timestamps

    async def _add_word_timestamps(self, word_times, context_id=None, **kwargs):
        for word_time in word_times:
            self.reported.append((context_id, *word_time))

    async def receive(self, message, context_id):
        await self.service._handle_alignment(ormsgpack.unpackb(message), context_id)

    def words(self):
        return [(context_id, word) for context_id, word, start, end in self.reported]


@pytest.mark.asyncio
async def test_a_word_is_reported_once_a_later_word_follows_it():
    recorder = _Recorder()

    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2)]), "ctx")
    assert recorder.reported == []

    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5)]), "ctx")
    assert recorder.reported == [("ctx", "Hi", 0.0, 0.2)]


@pytest.mark.asyncio
async def test_the_next_chunk_reports_the_last_word_of_the_one_before():
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5)]), "ctx")

    await recorder.receive(_audio(1, 0.8, [("Bye", 0.1, 0.4)]), "ctx")

    assert recorder.reported == [
        ("ctx", "Hi", 0.0, 0.2),
        ("ctx", "there", 0.2, 0.5),
    ]


@pytest.mark.asyncio
async def test_words_are_timed_from_the_start_of_their_context():
    # Two replies on one Fish session: the second starts 3 seconds into its audio.
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2)]), "first")
    await recorder.receive(_audio(1, 3.0, [("Bye", 0.5, 0.8)]), "second")

    await recorder.service._finish_all_chunks()

    assert recorder.reported == [
        ("first", "Hi", 0.0, 0.2),
        ("second", "Bye", 0.5, 0.8),
    ]


@pytest.mark.asyncio
async def test_the_repeated_last_chunk_counts_for_its_own_reply():
    # The first audio of the second reply carries the first reply's last alignment.
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5)]), "first")
    await recorder.service._finish_all_chunks()

    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5)]), "second")
    await recorder.receive(_audio(1, 0.6, [("Bye", 0.1, 0.4), ("now", 0.4, 0.6)]), "second")

    assert recorder.words() == [
        ("first", "Hi"),
        ("first", "there"),
        ("second", "Bye"),
    ]
    assert recorder.reported[-1] == ("second", "Bye", 0.1, 0.4)


@pytest.mark.asyncio
async def test_words_after_a_settled_chunk_are_still_reported():
    # The settle window passed mid-chunk, then the rest of the chunk arrived.
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("I", 0.0, 0.1)]), "ctx")
    await recorder.service._report_chunk(recorder.service._chunks[0], final=True)

    words = [("I", 0.0, 0.1), ("am", 0.1, 0.3), ("here", 0.3, 0.6)]
    await recorder.receive(_audio(0, 0.0, words), "ctx")
    await recorder.service._finish_all_chunks()

    assert recorder.words() == [("ctx", "I"), ("ctx", "am"), ("ctx", "here")]


@pytest.mark.asyncio
async def test_an_interrupted_context_reports_no_more_words():
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5)]), "ctx")

    await recorder.service.on_audio_context_interrupted("ctx")
    await recorder.receive(
        _audio(0, 0.0, [("Hi", 0.0, 0.2), ("there", 0.2, 0.5), ("you", 0.5, 0.7)]), "ctx"
    )
    await recorder.service._finish_all_chunks()

    assert recorder.words() == [("ctx", "Hi")]


@pytest.mark.asyncio
async def test_a_new_session_reports_its_chunks_from_the_start():
    recorder = _Recorder()
    await recorder.receive(_audio(3, 0.0, [("A", 0.0, 0.1)]), "ctx")
    await recorder.service._finish_all_chunks()

    recorder.service._reset_alignment()
    await recorder.receive(_audio(0, 0.0, [("B", 0.0, 0.1)]), "ctx")
    await recorder.service._finish_all_chunks()

    assert recorder.words() == [("ctx", "A"), ("ctx", "B")]


@pytest.mark.asyncio
async def test_a_quiet_chunk_reports_its_last_word(monkeypatch):
    monkeypatch.setattr(fish_tts, "ALIGNMENT_SETTLE_SECS", 0.01)
    recorder = _Recorder()
    await recorder.receive(_audio(0, 0.0, [("Hi", 0.0, 0.2)]), "ctx")

    settle = asyncio.create_task(recorder.service._alignment_settle_handler())
    await asyncio.sleep(0.05)
    settle.cancel()

    assert recorder.words() == [("ctx", "Hi")]


@pytest.mark.asyncio
async def test_malformed_segments_are_skipped():
    recorder = _Recorder()
    message = ormsgpack.packb(
        {
            "event": "audio",
            "audio": b"",
            "alignment": {"segments": [{"text": "Hi"}, {"text": "ok", "start": 0.1, "end": 0.3}]},
            "chunk_seq": 0,
            "chunk_audio_offset_sec": 0.0,
        }
    )

    await recorder.service._handle_alignment(ormsgpack.unpackb(message), "ctx")
    await recorder.service._finish_all_chunks()

    assert recorder.words() == [("ctx", "ok")]


@pytest.mark.asyncio
async def test_sent_tags_are_reported_before_the_word_after_them():
    recorder = _Recorder()
    sent = fish_tts._SentText()
    sent.add("[laugh] [happy] I am <slow> so happy </slow> for you. [short pause]")
    recorder.service._context_text["ctx"] = sent
    words = [
        ("I", 1.2, 1.4),
        ("am", 1.4, 1.6),
        ("so", 1.6, 2.2),
        ("happy", 2.2, 2.9),
        ("for", 2.9, 3.0),
        ("you", 3.0, 3.5),
    ]

    await recorder.receive(_audio(0, 0.0, words), "ctx")
    await recorder.service._finish_all_chunks()

    assert recorder.reported[:3] == [
        ("ctx", "[laugh]", 0.0, 0.0),
        ("ctx", "[happy]", 0.0, 0.0),
        ("ctx", "I", 1.2, 1.4),
    ]
    assert [word for _, word, _, _ in recorder.reported[3:]] == ["am", "so", "happy", "for", "you"]


def test_sent_text_follows_a_word_fish_splits_or_skips():
    sent = fish_tts._SentText()
    sent.add("Say MOO-ee-toh [laugh] now.")

    assert sent.tags_before("Say") == []
    assert sent.tags_before("MOO") == []
    assert sent.tags_before("ee") == []
    assert sent.tags_before("toh") == []
    assert sent.tags_before("unknown") == []
    assert sent.tags_before("now") == ["[laugh]"]


def test_without_word_timestamps_the_plain_endpoint_reports_no_words():
    service = FishAudioTTSService(api_key="key", word_timestamps=False)

    assert service._context_text == {}
    assert service._push_text_frames is True


async def _context_after_speaking(adapter, llm_text):
    """Format llm_text as the service sends it, report Fish's words for it, return the context text."""
    from pipecat.expressive.formatter import EXPRESSION_TAG_PATTERN, ExpressionFormatter
    from pipecat.frames.frames import AggregatedTextFrame, TTSTextFrame
    from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer

    async def transform(text, text_type):
        return text

    def trim(text):
        text = text.strip()
        return text or None

    formatter = ExpressionFormatter(lambda: adapter, "fish")
    segments = formatter.parse(llm_text)
    shown = ExpressionFormatter.shown_text(segments, "sentence")
    request = await formatter.format(segments, "sentence", transform, trim)
    sent_text = request.text

    sequencer = AggregatedFrameSequencer()
    sequencer.llm_markup = EXPRESSION_TAG_PATTERN
    frame = AggregatedTextFrame(shown, "sentence", raw_text=llm_text)
    await sequencer.register_spoken(
        frame, "ctx", sent_text, append_to_context=True, build_tracker=True
    )

    # Fish reports the words it spoke, without tags or punctuation.
    tracked = fish_tts._SentText()
    tracked.add(sent_text)
    without_tags = fish_tts._TEXT_TOKEN_RE.findall(sent_text)
    spoken = []
    for token in without_tags:
        if token.startswith("[") or token.startswith("<"):
            continue
        word = token.strip(".,!?;:")
        if word:
            spoken.append(word)

    context = []
    pts = 1
    for word in spoken:
        reported = [*tracked.tags_before(word), word]
        for item in reported:
            for out in sequencer.process_word(item, pts, "ctx"):
                if isinstance(out, TTSTextFrame) and out.append_to_context:
                    context.append(out.raw_text if out.raw_text is not None else out.text)
            pts += 1
    joined = " ".join(context)
    return " ".join(joined.split()), sent_text


_TRACKER_QUIRK = (
    "Pipecat's word tracker gives a re-stated delivery tag, or S2's [emphasis], a "
    "piece of the LLM's text; ElevenLabs' adapter shows the same"
)


def _context_cases():
    texts = [
        "<<excited>> <<whispering: Go on.>>",
        "<<sad>> <<fast: Go.>>",
        "<<excited>> Hey! [[laugh]] <<whispering: go on>> now.",
        "<<sad>> I'm fine. ^^Really.^^",
        "I'm fine. [[sigh]] Really.",
        "<<excited>> Great news! [[laugh]] It works. [[pause 500]]",
        "<<calm>> <<slow: Muito obrigado.>> It means thank you.",
        "Sure, <<excited>> it worked!",
        "<<happy>> Hey there! [[chuckle]] What's up?",
        "[[laugh]] <<excited>> No way!",
    ]
    quirks = {
        "<<excited>> Hey! [[laugh]] <<whispering: go on>> now.",
        "<<sad>> I'm fine. ^^Really.^^",
    }
    cases = []
    for adapter_class in (FishAudioDrama3ExpressionAdapter, FishAudioS2ExpressionAdapter):
        for text in texts:
            marks = []
            if adapter_class is FishAudioS2ExpressionAdapter and text in quirks:
                marks.append(pytest.mark.xfail(reason=_TRACKER_QUIRK, strict=True))
            cases.append(pytest.param(adapter_class, text, marks=marks))
    return cases


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_class, llm_text", _context_cases())
async def test_reported_words_rebuild_the_context_as_the_llm_wrote_it(adapter_class, llm_text):
    context, sent_text = await _context_after_speaking(adapter_class(), llm_text)

    expected = llm_text
    # A pause or sound at the very end is markup no word follows; the context
    # gets it when the context completes.
    for trailing in (" [[pause 500]]",):
        if expected.endswith(trailing) and not context.endswith(trailing):
            expected = expected[: -len(trailing)]
    assert context == expected, f"sent {sent_text!r}"
