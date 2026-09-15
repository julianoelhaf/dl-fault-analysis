"""Observability selection for the EPSR reduced-observability FL revision.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md for exact source commit and integration
notes. Manuscript Section 3.5 / Table 8.

This module maps a *faulted line* to the feature (channel) indices that a
reduced-observability model is allowed to see:

* ``full``           -- all measurement points (centralized 8-relay baseline).
* ``terminal_pair``  -- the two measurement points at the ends of the faulted
                        line (line-conditioned; 12 channels).
* ``single_ended``   -- one terminal of the faulted line (line-conditioned;
                        6 channels). Side ``"a"`` = the lower-bus end
                        (``line_from``), side ``"b"`` = the higher-bus end
                        (``line_to``).

The relay -> line -> channel mapping is derived **by parsing
``meta["feature_names"]``** (e.g. ``"Bus_1_Line_01_02A_cur_L1_A"``), which is
the authoritative, self-describing source -- this repository's own PROTECT-90
feature-name grammar, not a hardcoded channel-index table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

CHANNELS_PER_POINT = 6  # 3 current + 3 voltage per measurement point (relay)

ObservabilityMode = Literal[
    "full", "full_line_conditioned", "terminal_pair", "single_ended"
]
TerminalSide = Literal["a", "b"]

# Feature name, e.g. "Bus_1_Line_01_02A_cur_L1_A"
_FEATURE_RE = re.compile(
    r"^Bus_(?P<bus>\d+)_Line_(?P<frm>\d+)_(?P<to>\d+)(?P<circuit>[A-Za-z])"
    r"_(?P<qty>cur|vol)_L(?P<phase>\d)_(?P<unit>[A-Za-z]+)$"
)

# Faulted-line label, e.g. "Line_1_2_a" (label form) or "Line_01_02A" (feature form)
_LINE_RE = re.compile(r"^Line_(?P<frm>\d+)_(?P<to>\d+)_?(?P<circuit>[A-Za-z])$")


@dataclass(frozen=True)
class LineCircuit:
    """A canonical line-circuit identifier (parallel circuit of a corridor)."""

    line_from: int
    line_to: int
    circuit: str  # upper-case "A" / "B"

    def __str__(self) -> str:
        return f"Line_{self.line_from:02d}_{self.line_to:02d}{self.circuit}"


@dataclass(frozen=True)
class MeasurementPoint:
    """One relay/measurement point: a bus observing one line circuit (6 channels)."""

    bus: int
    circuit: LineCircuit
    channel_indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"Bus_{self.bus}_{self.circuit}"


@dataclass(frozen=True)
class FeatureLayout:
    """Parsed view of ``feature_names`` as ordered measurement points."""

    n_features: int
    points: tuple[MeasurementPoint, ...]

    def points_for_circuit(self, circuit: LineCircuit) -> List[MeasurementPoint]:
        """The measurement points observing ``circuit``, ordered from-bus then to-bus."""
        matches = [p for p in self.points if p.circuit == circuit]
        return sorted(matches, key=lambda p: p.bus)


def normalize_line_token(line: str) -> LineCircuit:
    """Normalize a faulted-line string to a canonical :class:`LineCircuit`.

    Accepts both label form (``"Line_1_2_a"``) and feature form (``"Line_01_02A"``).
    Raises ``ValueError`` on anything that is not a parseable line id.
    """
    if not isinstance(line, str):
        raise ValueError(f"faulted line must be a string, got {type(line)!r}")
    m = _LINE_RE.match(line.strip())
    if not m:
        raise ValueError(
            f"Unrecognized faulted-line token {line!r} "
            f"(expected e.g. 'Line_1_2_a' or 'Line_01_02A')"
        )
    return LineCircuit(
        line_from=int(m.group("frm")),
        line_to=int(m.group("to")),
        circuit=m.group("circuit").upper(),
    )


def parse_feature_layout(feature_names: Sequence[str]) -> FeatureLayout:
    """Parse ``feature_names`` into ordered :class:`MeasurementPoint`s.

    Groups channels by ``(bus, line circuit)``. Validates that every point has
    exactly ``CHANNELS_PER_POINT`` channels. Fails loudly on missing/empty
    metadata or unparseable names.
    """
    if not feature_names:
        raise ValueError("feature_names is empty/missing; cannot derive observability")

    # Preserve first-seen order of channels per measurement point.
    grouped: Dict[tuple[int, LineCircuit], List[int]] = {}
    order: List[tuple[int, LineCircuit]] = []
    for idx, name in enumerate(feature_names):
        m = _FEATURE_RE.match(str(name).strip())
        if not m:
            raise ValueError(
                f"feature name {name!r} (index {idx}) does not match the expected "
                "'Bus_<b>_Line_<f>_<t><A|B>_<cur|vol>_L<p>_<unit>' pattern"
            )
        circuit = LineCircuit(
            line_from=int(m.group("frm")),
            line_to=int(m.group("to")),
            circuit=m.group("circuit").upper(),
        )
        key = (int(m.group("bus")), circuit)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(idx)

    points: List[MeasurementPoint] = []
    for bus, circuit in order:
        chans = grouped[(bus, circuit)]
        if len(chans) != CHANNELS_PER_POINT:
            raise ValueError(
                f"measurement point Bus_{bus}_{circuit} has {len(chans)} channels, "
                f"expected {CHANNELS_PER_POINT}"
            )
        points.append(
            MeasurementPoint(bus=bus, circuit=circuit, channel_indices=tuple(chans))
        )

    return FeatureLayout(n_features=len(feature_names), points=tuple(points))


def get_relay_mapping(feature_names: Sequence[str]) -> Dict[str, tuple[int, ...]]:
    """Return ``{measurement_point_label: channel_indices}`` for all points."""
    layout = parse_feature_layout(feature_names)
    return {p.label: p.channel_indices for p in layout.points}


def get_relays_for_line(
    feature_names: Sequence[str], faulted_line: str
) -> Dict[str, MeasurementPoint]:
    """Return the two terminal measurement points of ``faulted_line``.

    Keys are ``"a"`` (from-bus end) and ``"b"`` (to-bus end). Raises if the line
    is unknown or is not observed at exactly two terminals.
    """
    layout = parse_feature_layout(feature_names)
    circuit = normalize_line_token(faulted_line)
    points = layout.points_for_circuit(circuit)
    if not points:
        known = sorted({str(p.circuit) for p in layout.points})
        raise ValueError(
            f"faulted line {faulted_line!r} (={circuit}) not found in feature layout; "
            f"known circuits: {known}"
        )
    if len(points) != 2:
        raise ValueError(
            f"line {circuit} is observed at {len(points)} measurement points "
            f"({[p.label for p in points]}); terminal-pair requires exactly 2"
        )
    # points are sorted by bus; lower bus is the 'a' (from) side.
    return {"a": points[0], "b": points[1]}


def select_observability(
    feature_names: Sequence[str],
    mode: ObservabilityMode,
    faulted_line: str | None = None,
    terminal_side: TerminalSide | None = None,
) -> List[int]:
    """Return the sorted feature indices visible under ``mode``.

    * ``full`` / ``full_line_conditioned``: all indices (``faulted_line`` ignored;
      the two differ only in the runner -- ``full`` is one global model, whereas
      ``full_line_conditioned`` trains a per-line model on all 8 relays, the
      control that isolates the per-line-model effect from the relay-count effect).
    * ``terminal_pair``: both terminals of ``faulted_line`` (requires ``faulted_line``).
    * ``single_ended``: one terminal of ``faulted_line`` selected by ``terminal_side``
      (``"a"`` = from-bus, ``"b"`` = to-bus); both required.
    """
    layout = parse_feature_layout(feature_names)

    if mode in ("full", "full_line_conditioned"):
        return list(range(layout.n_features))

    if mode in ("terminal_pair", "single_ended"):
        if faulted_line is None:
            raise ValueError(f"mode={mode!r} requires a faulted_line")
        terminals = get_relays_for_line(feature_names, faulted_line)
        if mode == "terminal_pair":
            idx = list(terminals["a"].channel_indices) + list(
                terminals["b"].channel_indices
            )
            return sorted(idx)
        # single_ended
        if terminal_side not in ("a", "b"):
            raise ValueError(
                f"mode='single_ended' requires terminal_side in {{'a','b'}}, "
                f"got {terminal_side!r}"
            )
        return sorted(terminals[terminal_side].channel_indices)

    raise ValueError(
        f"unknown observability mode {mode!r}; expected one of "
        "'full', 'full_line_conditioned', 'terminal_pair', 'single_ended'"
    )


def expected_channel_count(mode: ObservabilityMode, n_features: int) -> int:
    """Number of channels a valid selection must have for ``mode``."""
    if mode in ("full", "full_line_conditioned"):
        return n_features
    if mode == "terminal_pair":
        return 2 * CHANNELS_PER_POINT
    if mode == "single_ended":
        return CHANNELS_PER_POINT
    raise ValueError(f"unknown observability mode {mode!r}")


def validate_observability_selection(
    selected_indices: Sequence[int],
    feature_names: Sequence[str],
    mode: ObservabilityMode,
) -> None:
    """Fail loudly if ``selected_indices`` is malformed for ``mode``.

    Checks: non-empty, in-bounds, unique, and the expected channel count.
    """
    n = len(feature_names)
    if len(selected_indices) == 0:
        raise ValueError(f"empty observability selection for mode={mode!r}")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError(
            f"duplicate indices in observability selection: {selected_indices}"
        )
    if min(selected_indices) < 0 or max(selected_indices) >= n:
        raise ValueError(
            f"observability indices out of bounds [0,{n - 1}]: {selected_indices}"
        )
    expected = expected_channel_count(mode, n)
    if len(selected_indices) != expected:
        raise ValueError(
            f"mode={mode!r} expected {expected} channels, got {len(selected_indices)}"
        )


def describe_selection(
    feature_names: Sequence[str],
    mode: ObservabilityMode,
    faulted_line: str | None = None,
    terminal_side: TerminalSide | None = None,
) -> Dict[str, object]:
    """Human/JSON-friendly description of a selection (for outputs/tables)."""
    indices = select_observability(feature_names, mode, faulted_line, terminal_side)
    layout = parse_feature_layout(feature_names)
    chosen = set(indices)
    relays = [p.label for p in layout.points if set(p.channel_indices) <= chosen]
    return {
        "mode": mode,
        "faulted_line": faulted_line,
        "terminal_side": terminal_side,
        "n_relays": len(relays),
        "selected_relays": relays,
        "n_channels": len(indices),
        "feature_indices": indices,
    }
