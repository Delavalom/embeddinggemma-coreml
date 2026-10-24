from scripts.check_gates_adr import violates


def test_gates_change_needs_adr():
    assert violates(["gates.toml"]) is True
    assert violates(["gates.toml", "docs/decisions/ADR-0005-x.md"]) is False
    assert violates(["egcm/metrics.py"]) is False


def test_adr_elsewhere_does_not_count():
    assert violates(["gates.toml", "docs/ADR-0007-x.md"]) is True
    assert violates(["gates.toml", "stages/s5_swift/docs/decisions/ADR-0007.md"]) is True


def test_only_root_gates_file_counts():
    assert violates(["stages/s4_compress/gates.toml"]) is False
    assert violates([]) is False
