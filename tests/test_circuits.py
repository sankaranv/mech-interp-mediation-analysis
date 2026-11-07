import json
from pathlib import Path

from mediation.circuits import (
    IOI_ROLE_COLORS,
    IOI_ROLE_ORDER,
    ioi_circuit_heads,
    ioi_role,
)
from mediation.patching import AttentionHead

WANG_CIRCUIT = (
    Path(__file__).resolve().parents[1] / "data" / "ioi" / "wang_circuit.json"
)


def test_circuit_heads_match_the_shipped_circuit_file():
    records = json.loads(WANG_CIRCUIT.read_text())
    shipped = sorted(AttentionHead(r["layer"], r["head_or_mlp"]) for r in records)
    assert ioi_circuit_heads() == shipped
    assert len(shipped) == 26


def test_every_role_has_a_colour_and_a_place_in_the_legend():
    assert set(IOI_ROLE_ORDER) == set(IOI_ROLE_COLORS)
    assert ioi_role(AttentionHead(9, 9)) == "Name Mover"
    assert ioi_role(AttentionHead(0, 0)) == "Other"
