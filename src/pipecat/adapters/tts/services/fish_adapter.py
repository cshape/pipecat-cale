#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter, delivery_tags
from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The tag each speed delivery is written as.
_SPEED_TAGS = {SLOW_DELIVERY: "speaking slowly", FAST_DELIVERY: "speaking quickly"}
# The tag that sets the pace back. A pace a tag sets lasts past the input it
# is sent in, until another tag changes it.
_NORMAL_PACE_TAG = "[normal pace]"

# The tags in Fish Audio's S2 emotion reference, sorted into sounds made at one
# point and deliveries that last. Multi-word deliveries, such as "soft tone",
# aren't listed; written in a tag, they're passed on as free text.
_S2_SOUNDS = [
    ExpressionTerm("laughing"),
    ExpressionTerm("chuckling"),
    ExpressionTerm("sobbing"),
    ExpressionTerm("crying loudly"),
    ExpressionTerm("sighing"),
    ExpressionTerm("groaning"),
    ExpressionTerm("panting"),
    ExpressionTerm("gasping"),
    ExpressionTerm("yawning"),
    ExpressionTerm("snoring"),
    ExpressionTerm("clear throat"),
    ExpressionTerm("audience laughing", "sound effect"),
    ExpressionTerm("background laughter", "sound effect"),
    ExpressionTerm("crowd laughing", "sound effect"),
]

_S2_DELIVERY = [
    ExpressionTerm(name)
    for name in (
        "happy sad angry excited calm nervous confident surprised satisfied delighted "
        "scared worried upset frustrated depressed empathetic embarrassed disgusted moved "
        "proud relaxed grateful curious sarcastic disdainful unhappy anxious hysterical "
        "indifferent uncertain doubtful confused disappointed regretful guilty ashamed "
        "jealous envious hopeful optimistic pessimistic nostalgic lonely bored contemptuous "
        "sympathetic compassionate determined resigned shouting screaming whispering"
    ).split()
]


class FishAudioS2ExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Fish Audio S2 bracket tags.

    Part of expressive conversation (alpha). Delivery and sounds are both tags,
    such as ``[excited]`` or ``[laughing]``, written in free text: the terms
    listed are Fish Audio's S2 emotion reference, and others described in a
    few words work too. A tag applies within the input it is sent in, so each
    sentence states its delivery from its start. There is no tag for the voice's
    baseline, so a return to it writes none. Slow and fast are written as
    ``[speaking slowly]`` and ``[speaking quickly]``, and since a pace lasts
    until another tag changes it, a ``[normal pace]`` tag ends them, within a
    sentence or at the end of one. Emphasis is an ``[emphasis]`` tag
    before the stressed words, and a pause is a break or a long break, which
    is approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            sounds=list(_S2_SOUNDS),
            delivery=[*_S2_DELIVERY, *SPEED_DELIVERY],
            allows_free_text_sounds=True,
            allows_free_text_delivery=True,
            pause_range_ms=(100, 1500),
            emphasis=True,
            service_vocabulary={
                ExpressionDelivery.CALM: ["calm", "relaxed"],
                ExpressionDelivery.SYMPATHETIC: ["sympathetic", "empathetic", "compassionate"],
                ExpressionDelivery.APOLOGETIC: ["regretful"],
                ExpressionDelivery.WARM: ["happy", "relaxed"],
                ExpressionDelivery.EXCITED: ["excited", "delighted"],
                ExpressionSound.LAUGH: ["laughing"],
                ExpressionSound.CHUCKLE: ["chuckling"],
                ExpressionSound.SIGH: ["sighing"],
                ExpressionSound.BREATH: ["inhaling"],
                ExpressionSound.LIGHT_BREATH: ["inhaling"],
                ExpressionSound.EXHALE: ["exhaling"],
                ExpressionSound.SOB: ["sobbing", "crying loudly"],
                ExpressionSound.GROAN: ["groaning"],
                ExpressionSound.PANT: ["panting"],
                ExpressionSound.GASP: ["gasping"],
                ExpressionSound.YAWN: ["yawning"],
                ExpressionSound.SNORE: ["snoring"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix tags where the delivery changes, and emphasis before stressed words.

        Each tag is set apart by spaces, so word timestamps line up with the words.
        """
        tags = delivery_tags(delivery, _SPEED_TAGS)
        previous = delivery_tags(previous_delivery, _SPEED_TAGS)
        prefix_tags = []
        if _has_speed(previous_delivery or []) and not _has_speed(delivery):
            prefix_tags.append(_NORMAL_PACE_TAG)
        if tags and tags != previous:
            for tag in tags:
                prefix_tags.append(f"[{tag}]")
        if EMPHASIS_DELIVERY in delivery:
            prefix_tags.append("[emphasis]")
        if not prefix_tags:
            return text
        prefix = " ".join(prefix_tags)
        return f" {prefix} {text}"

    def format_end(self, delivery: list[str]) -> str:
        """Set the pace back after a request that ends in a speed delivery."""
        return f" {_NORMAL_PACE_TAG}" if _has_speed(delivery) else ""

    def format_sound(self, name: str) -> str:
        """Format a tag, set apart from the words."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a break under a second, otherwise a long break, set apart from the words."""
        if duration_ms < 1000:
            return " [break] "
        return " [long-break] "


def _has_speed(delivery: list[str]) -> bool:
    return any(name in _SPEED_TAGS for name in delivery)


# Deliveries Drama 3 performs as a paired tag around the words, by its tag name.
# A paired tag changes only the words it wraps.
_DRAMA3_PAIRED_TAGS = {
    "whispering": "whisper",
    "soft": "soft",
    "forceful": "forceful",
    SLOW_DELIVERY: "slow",
    FAST_DELIVERY: "fast",
}

# Sounds from Drama 3's direction guide, written in brackets where they happen.
# Others described in a few words work too.
_DRAMA3_SOUNDS = [
    ExpressionTerm("laugh"),
    ExpressionTerm("small laugh"),
    ExpressionTerm("chuckle"),
    ExpressionTerm("giggle"),
    ExpressionTerm("sigh"),
    ExpressionTerm("gasp"),
    ExpressionTerm("breath"),
    ExpressionTerm("deep breath"),
    ExpressionTerm("exhale"),
    ExpressionTerm("sob"),
    ExpressionTerm("groan"),
    ExpressionTerm("yawn"),
    ExpressionTerm("clears throat"),
]

# Directions are free text; these are the one-word ones offered by name.
_DRAMA3_DIRECTIONS = [
    ExpressionTerm(name)
    for name in (
        "happy sad angry excited calm warm nervous confident surprised delighted scared "
        "worried upset frustrated empathetic sympathetic compassionate embarrassed proud "
        "relaxed grateful curious playful sarcastic anxious hopeful disappointed "
        "reassuring relieved apologetic determined shouting"
    ).split()
]

_DRAMA3_PAIRED_DELIVERY = [
    ExpressionTerm("whispering"),
    ExpressionTerm("soft"),
    ExpressionTerm("forceful"),
    *SPEED_DELIVERY,
]


class FishAudioDrama3ExpressionAdapter(BaseExpressionAdapter):
    """Formats expression for Fish Audio's Drama 3 model.

    Part of expressive conversation (alpha). Drama 3 is directed in plain
    language: a direction in square brackets, such as ``[excited]``, sets how
    the words after it are performed, and a sound or pause in square brackets,
    such as ``[sigh]`` or ``[short pause]``, happens where it is written. Both
    take free text. A delivery that applies to some words only is a paired tag
    around them: whispering, soft, forceful, slow and fast are written as
    ``<whisper>``, ``<soft>``, ``<forceful>``, ``<slow>`` and ``<fast>``, and
    emphasis as ``<emphasis>``. Tags are set apart from the words by spaces.
    There is no direction for the voice's baseline, so a return to it writes
    none.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text directions and sounds, word-level paired tags, emphasis and pauses."""
        return ExpressionCapabilities(
            sounds=list(_DRAMA3_SOUNDS),
            delivery=[*_DRAMA3_DIRECTIONS, *_DRAMA3_PAIRED_DELIVERY],
            allows_free_text_sounds=True,
            allows_free_text_delivery=True,
            pause_range_ms=(100, 2000),
            emphasis=True,
            service_vocabulary={
                ExpressionDelivery.CALM: ["calm", "relaxed"],
                ExpressionDelivery.SYMPATHETIC: ["sympathetic", "empathetic", "compassionate"],
                ExpressionDelivery.APOLOGETIC: ["apologetic"],
                ExpressionDelivery.WARM: ["warm"],
                ExpressionDelivery.EXCITED: ["excited", "delighted"],
                ExpressionDelivery.WHISPERING: ["whispering"],
                ExpressionDelivery.LOUD: ["forceful"],
                ExpressionSound.LAUGH: ["laugh"],
                ExpressionSound.CHUCKLE: ["chuckle", "small laugh"],
                ExpressionSound.GIGGLE: ["giggle"],
                ExpressionSound.SIGH: ["sigh"],
                ExpressionSound.BREATH: ["breath"],
                ExpressionSound.LIGHT_BREATH: ["breath"],
                ExpressionSound.DEEP_BREATH: ["deep breath"],
                ExpressionSound.EXHALE: ["exhale"],
                ExpressionSound.SOB: ["sob"],
                ExpressionSound.GROAN: ["groan"],
                ExpressionSound.GASP: ["gasp"],
                ExpressionSound.YAWN: ["yawn"],
                ExpressionSound.CLEAR_THROAT: ["clears throat"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Write a direction where it changes, and wrap the words in their paired tags."""
        directions = _drama3_directions(delivery)
        previous_directions = _drama3_directions(previous_delivery or [])

        body = text
        for name in delivery:
            tag = _DRAMA3_PAIRED_TAGS.get(name)
            if tag is not None:
                body = f"<{tag}> {body} </{tag}>"
        if EMPHASIS_DELIVERY in delivery:
            body = f"<emphasis> {body} </emphasis>"

        if directions and directions != previous_directions:
            direction_text = ", ".join(directions)
            return f" [{direction_text}] {body}"
        return body

    def format_sound(self, name: str) -> str:
        """Format a sound in brackets where it happens, set apart from the words."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a short pause under a second, otherwise a long one, which is approximate."""
        if duration_ms < 1000:
            return " [short pause] "
        return " [long pause] "


def _drama3_directions(delivery: list[str]) -> list[str]:
    """The names in a delivery that Drama 3 performs as a direction, not a paired tag."""
    directions = []
    for name in delivery:
        if name in _DRAMA3_PAIRED_TAGS:
            continue
        if name == EMPHASIS_DELIVERY:
            continue
        directions.append(name)
    return directions
