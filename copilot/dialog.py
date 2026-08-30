"""Parsing customer turns into structured constraints, and the session state.

The simulator speaks in a small set of templates.  We parse them exactly when we
can and fall back to treating the whole utterance as free text when we cannot,
so that added paraphrasing degrades the signal rather than breaking the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

BROWSING_TAIL = ", but I'm still exploring."
BUYING_JOIN = ". A key requirement is: "
LOOKING_PREFIX = "I'm looking for "

RE_REPLY = re.compile(r"^For that, what matters is:\s*(?P<body>.+?)\.?$", re.S)
RE_OVERRIDE = re.compile(
    r"^Actually, ignore my earlier preference\.\s*What I need is:\s*(?P<body>.+?)\.?$", re.S
)
RE_NO_EXTRA = re.compile(r"^I don't have an additional preference for\s+(?P<attr>[\w_]+)\.?$")
RE_BOUNDARY = re.compile(
    r"^I don't have a preference for\s+(?P<attr>[\w_]+);\s*please use your judgment\.?$"
)
RE_NUDGE = re.compile(r"^Those options are not quite right yet\.")


@dataclass
class Constraint:
    text: str
    known: bool = False
    emphasized: bool = False
    mined: bool = False


@dataclass(frozen=True)
class AskEvidence:
    """One question of ours and the answer it actually produced.

    Enough to replay the exchange against any candidate: see `copilot.replay`.
    `disclosed_before` is a snapshot taken *before* this turn's own disclosures,
    because that is the state the simulator was in when it chose its answer.
    """

    attribute: str
    revealed: tuple[str, ...]
    disclosed_before: frozenset[str]


@dataclass
class SessionState:
    session_id: str
    profile: dict = field(default_factory=dict)
    category: str | None = None
    scenario: str = "unknown"
    constraints: list[Constraint] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    dead_attributes: set[str] = field(default_factory=set)
    boundary_seen: bool = False
    override_seen: bool = False
    transcript: list[str] = field(default_factory=list)
    last_recommendations: list[str] = field(default_factory=list)

    disclosed: set[str] = field(default_factory=set)
    """Mirror of the simulator's own `disclosed` set.

    Deliberately *not* `seen`: that also holds mined guesses and unparsed free
    text, none of which the simulator considers disclosed.  A false entry here
    would make the replay skip a real card constraint and rule out the target.
    """

    evidence: list[AskEvidence] = field(default_factory=list)

    parse_failures: int = 0
    """Turns we could not read as any known template.

    A health signal for the strategies that assume we are following the
    conversation. Non-zero means the customer is being worded in a way we do not
    recognise, and anything premised on "one more question will settle this" no
    longer holds.
    """

    def add_constraint(
        self, text: str, known: bool, emphasized: bool = False, mined: bool = False
    ) -> None:
        text = text.strip()
        if not text or text in self.seen:
            if text in self.seen and emphasized:
                for item in self.constraints:
                    if item.text == text:
                        item.emphasized = True
            return
        self.seen.add(text)
        self.constraints.append(
            Constraint(text=text, known=known, emphasized=emphasized, mined=mined)
        )

    @property
    def preference_tags(self) -> list[str]:
        tags = self.profile.get("preference_tags")
        return [str(t) for t in tags] if isinstance(tags, list) else []

    def dialog_text(self) -> str:
        return " ".join(self.transcript)


class DialogParser:
    """Turns simulator utterances into constraints, validated against the index."""

    def __init__(self, index) -> None:
        self.index = index
        self._categories = set(index.category_index)

    # ------------------------------------------------------------- utilities

    def _known(self, text: str) -> bool:
        return text in self.index.card_index or text in self.index.loose_index

    def _segment(self, body: str) -> list[tuple[str, bool]]:
        """Split a reply body into constraints.

        The simulator joins at most two constraints with "; ", so there is at
        most one true split point.  A constraint may itself contain "; ", which
        is why we validate each candidate split against the index instead of
        splitting blindly.
        """
        body = body.strip().rstrip(".")
        if not body:
            return []
        if self._known(body):
            return [(body, True)]

        best: list[tuple[str, bool]] | None = None
        for match in re.finditer(r";\s+", body):
            left, right = body[: match.start()], body[match.end():]
            if not left or not right:
                continue
            left_ok, right_ok = self._known(left), self._known(right)
            if left_ok and right_ok:
                return [(left, True), (right, True)]
            if best is None and (left_ok or right_ok):
                best = [(left, left_ok), (right, right_ok)]
        if best is not None:
            return best
        return [(body, False)]

    def _split_category(self, rest: str) -> tuple[str | None, str | None]:
        """Peel a known coarse-category off the front of an opening message."""
        for match in re.finditer(r"\.\s+", rest):
            head = rest[: match.start()]
            if head in self._categories:
                return head, rest[match.end():]
        # Fall back to the first sentence boundary even if unrecognised.
        match = re.search(r"\.\s+", rest)
        if match:
            return rest[: match.start()], rest[match.end():]
        return rest.rstrip("."), None

    # ---------------------------------------------------------------- public

    def ingest(
        self, state: SessionState, message: str, turn: int, config=None, llm=None
    ) -> None:
        text = (message or "").strip()
        state.transcript.append(text)
        if not text:
            return
        known_before = sum(1 for c in state.constraints if c.known)
        self._ingest_templates(state, text, turn)
        parsed_something = sum(1 for c in state.constraints if c.known) > known_before
        if not (parsed_something and getattr(config, "mining_only_when_parse_fails", False)):
            self._mine(state, text, config)
            if llm is not None and not parsed_something:
                self._llm_parse(state, text, config, llm)

    def _llm_parse(self, state: SessionState, text: str, config, llm) -> None:
        """Last resort when the templates missed: ask which constraint was meant.

        Only runs on turns the parser could not read, and only chooses among
        constraints that already exist in the catalog -- the model narrows a
        shortlist, it never introduces a string. A miss leaves the mined
        evidence exactly as it was.
        """
        shortlist = self.index.mine_constraints(
            text,
            min_overlap=0.25,
            limit=config.mining_candidates,
            max_results=16,
            min_tokens=config.mining_min_tokens,
        )
        if not shortlist:
            return
        chosen = llm.select_constraint(text, shortlist)
        if chosen and chosen in self.index.card_index:
            # Full weight: unlike a mined guess this one was adjudicated.
            state.add_constraint(chosen, known=True)

    def _mine(self, state: SessionState, text: str, config) -> None:
        """Template-independent salvage.

        Measured on the robustness harness, rewording the templates alone cost
        more score than paraphrasing the constraints inside them -- the parser
        was the weak link, not the matching.  Mining recovers constraints by
        token overlap, so it does not care how the turn is phrased.
        """
        if config is None or not getattr(config, "use_constraint_mining", False):
            return
        for candidate in self.index.mine_constraints(
            text,
            config.mining_min_overlap,
            config.mining_candidates,
            config.mining_max_results,
            config.mining_min_tokens,
        ):
            state.add_constraint(candidate, known=True, mined=True)

    def _ingest_templates(self, state: SessionState, text: str, turn: int) -> None:
        if turn == 1:
            self._ingest_opening(state, text)
            return

        # Whatever we asked last turn is what this message answers: `ingest` runs
        # before the turn's own question is chosen, so this is never stale.
        pending = state.asked[-1] if state.asked else None

        match = RE_OVERRIDE.match(text)
        if match:
            state.override_seen = True
            # The replacement value comes from the same target's intent card, so
            # earlier evidence stays valid -- we re-weight rather than erase.
            for item, known in self._segment(match.group("body")):
                state.add_constraint(item, known, emphasized=True)
                if known:
                    state.disclosed.add(item)
            # Not evidence: the evaluator injects this turn on a fixed schedule
            # instead of answering us, so it says nothing about our question.
            return

        match = RE_REPLY.match(text)
        if match:
            segments = self._segment(match.group("body"))
            before = frozenset(state.disclosed)
            for item, known in segments:
                state.add_constraint(item, known)
                if known:
                    state.disclosed.add(item)
            # A partial parse means we do not really know what was said, and a
            # wrong `revealed` would rule out the target.  Stay silent instead.
            if pending and segments and all(known for _, known in segments):
                state.evidence.append(
                    AskEvidence(pending, tuple(item for item, _ in segments), before)
                )
            return

        match = RE_BOUNDARY.match(text)
        if match:
            state.boundary_seen = True
            state.dead_attributes.add(match.group("attr"))
            # Carries no information: the evaluator fires this once per boundary
            # session whatever the target's card looks like.  Treating it as a
            # real refusal excluded the true target in 8% of turn-states.
            return

        match = RE_NO_EXTRA.match(text)
        if match:
            attribute = match.group("attr")
            state.dead_attributes.add(attribute)
            # This one *is* evidence: the target has no undisclosed constraint of
            # this class, which rules out every candidate that still has one.
            if state.asked:
                state.evidence.append(
                    AskEvidence(attribute, (), frozenset(state.disclosed))
                )
            return

        if RE_NUDGE.match(text):
            return

        # Unrecognised phrasing (e.g. organizer paraphrasing): keep it as a
        # free-text constraint so BM25 can still use it.
        state.parse_failures += 1
        state.add_constraint(text, known=False)

    def _ingest_opening(self, state: SessionState, text: str) -> None:
        if not text.startswith(LOOKING_PREFIX):
            state.scenario = "unknown"
            state.parse_failures += 1
            state.add_constraint(text, known=False)
            return

        rest = text[len(LOOKING_PREFIX):]

        if rest.endswith(BROWSING_TAIL):
            state.scenario = "browsing"
            state.category = rest[: -len(BROWSING_TAIL)].strip()
            return

        if BUYING_JOIN in rest:
            category, body = rest.split(BUYING_JOIN, 1)
            state.scenario = "buying"
            state.category = category.strip()
            for item, known in self._segment(body):
                state.add_constraint(item, known)
                if known:
                    state.disclosed.add(item)
            return

        category, remainder = self._split_category(rest)
        state.category = (category or "").strip() or None
        state.scenario = "intent_override"
        if remainder:
            for item, known in self._segment(remainder):
                state.add_constraint(item, known)
                # Deliberately not disclosed: the evaluator speaks this opening
                # value without adding it to its own `disclosed` set, so the
                # simulator may legitimately reveal it again later.

