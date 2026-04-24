from pathlib import Path

import pytest

from agentlab.suites import load_suite


def test_load_smoke_suite():
    suite = load_suite(Path(__file__).resolve().parent.parent / "examples" / "suites" / "smoke.yaml")
    assert suite.name == "smoke"
    assert len(suite.tasks) == 2
    ids = {t.id for t in suite.tasks}
    assert ids == {"say_hello", "arithmetic"}


def test_parametric_expansion():
    suite = load_suite(
        Path(__file__).resolve().parent.parent / "examples" / "suites" / "python_refactor.yaml"
    )
    ids = sorted(t.id for t in suite.tasks)
    assert ids == ["solve_addition_javascript", "solve_addition_python"]
    # Check that the prompt template was rendered differently.
    p = next(t for t in suite.tasks if t.id == "solve_addition_python")
    j = next(t for t in suite.tasks if t.id == "solve_addition_javascript")
    assert "python" in p.prompt.lower()
    assert "javascript" in j.prompt.lower()


def test_defaults_applied(tmp_path):
    yaml_text = """
name: t
version: "1"
defaults:
  timeout_s: 77
tasks:
  - id: a
    description: d
    prompt: p
"""
    f = tmp_path / "suite.yaml"
    f.write_text(yaml_text)
    suite = load_suite(f)
    assert suite.tasks[0].timeout_s == 77


def test_rejects_non_mapping_root(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_suite(f)
