"""The local .env loader."""

import os

from classic_patterns.utils.config import load_env


def test_load_env_parses_and_does_not_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n\nSEC_USER_AGENT=\"A B a@b.example\"\nexport PE_TEST_X='1 2'\n"
        "PE_TEST_SET=from_file\nnot a pair\n"
    )
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("PE_TEST_X", raising=False)
    monkeypatch.setenv("PE_TEST_SET", "from_shell")
    got = load_env(p)
    assert got == {
        "SEC_USER_AGENT": "A B a@b.example",
        "PE_TEST_X": "1 2",
        "PE_TEST_SET": "from_file",
    }
    assert os.environ["SEC_USER_AGENT"] == "A B a@b.example"
    assert os.environ["PE_TEST_X"] == "1 2"
    assert os.environ["PE_TEST_SET"] == "from_shell"  # the shell wins
    assert load_env(tmp_path / "missing") == {}
