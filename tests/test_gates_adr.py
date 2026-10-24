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


def test_main_checks_the_git_range(repo):
    from conftest import commit

    from scripts.check_gates_adr import main

    (repo / "gates.toml").write_text("[s0_reference]\nmin_a = 0.4\n")
    commit(repo, "loosen a gate")
    assert main(["HEAD~1...HEAD"], root=repo) == 1
    (repo / "docs" / "decisions").mkdir(parents=True)
    (repo / "docs" / "decisions" / "ADR-0009-why.md").write_text("# why\n")
    (repo / "gates.toml").write_text("[s0_reference]\nmin_a = 0.3\n")
    commit(repo, "gate change with its ADR")
    assert main(["HEAD~1...HEAD"], root=repo) == 0
    assert main(["HEAD~2...HEAD"], root=repo) == 0  # a range: the ADR anywhere in it counts
