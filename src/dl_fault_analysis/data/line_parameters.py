"""Per-episode true line parameters for the impedance-based FL baseline.

Ported from the private development repository's EPSR reviewer-response
branch (Reviewer #2.1) -- see docs/PROVENANCE.md. Manuscript Section 3.7 /
Table 10.

Loads the PROTECT-90 scenario metadata CSV
(``hv_double_line_90kv_labels.csv``), which records, per episode (``sample_id``),
the randomized series-impedance parameters of every line section plus the fault
descriptors. For the impedance baseline we only need the parameters of the
*faulted* line; ``faulted_line_params`` returns them as a typed record.

Column convention (per line, e.g. ``line_2_3_a``):
  ``*_length`` km, ``*_rline``/``*_xline`` positive-seq R'/X' (Ohm/km),
  ``*_cline`` positive-seq C' (uF/km), and ``*_rline0``/``*_xline0``/``*_cline0``
  the zero-sequence counterparts. ``sc_location`` is the fault position in
  **percent of line length** (identical to ``y_fault_location``), measured from
  the ``line_from`` terminal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Map the fault-line label (as in y_fault_line / fault_target) to the CSV prefix.
_LINE_PREFIX = {
    "Line_1_2_a": "line_1_2_a",
    "Line_1_2_b": "line_1_2_b",
    "Line_2_3_a": "line_2_3_a",
    "Line_2_3_b": "line_2_3_b",
}


@dataclass(frozen=True)
class LineParams:
    """True series parameters of one faulted line section (per episode)."""

    sample_id: int
    line: str
    length_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    r0_ohm_per_km: float
    x0_ohm_per_km: float

    @property
    def z1_total(self) -> complex:
        """Total positive-sequence series impedance (Ohm)."""
        return complex(self.r1_ohm_per_km, self.x1_ohm_per_km) * self.length_km

    @property
    def z0_total(self) -> complex:
        """Total zero-sequence series impedance (Ohm)."""
        return complex(self.r0_ohm_per_km, self.x0_ohm_per_km) * self.length_km


def load_line_parameters(csv_path: str) -> pd.DataFrame:
    """Load the scenario-metadata CSV indexed by ``sample_id``."""
    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'sample_id' column")
    return df.set_index("sample_id", drop=False)


def faulted_line_params(
    params_df: pd.DataFrame, sample_id: int, line: str
) -> LineParams:
    """Return the faulted line's true parameters for ``sample_id``.

    ``line`` is the fault-line label (e.g. ``Line_2_3_a``); raises on an unknown
    line or a missing episode so the baseline fails loudly rather than silently
    mislocating with wrong parameters.
    """
    if line not in _LINE_PREFIX:
        raise ValueError(f"unknown fault line {line!r}; expected {list(_LINE_PREFIX)}")
    if sample_id not in params_df.index:
        raise KeyError(f"sample_id {sample_id} not in line-parameter table")
    p = _LINE_PREFIX[line]
    row = params_df.loc[sample_id]
    return LineParams(
        sample_id=int(sample_id),
        line=line,
        length_km=float(row[f"{p}_length"]),
        r1_ohm_per_km=float(row[f"{p}_rline"]),
        x1_ohm_per_km=float(row[f"{p}_xline"]),
        r0_ohm_per_km=float(row[f"{p}_rline0"]),
        x0_ohm_per_km=float(row[f"{p}_xline0"]),
    )
