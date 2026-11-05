from classic_patterns.utils.code_hash import code_hash, tree_hash


def test_tree_hash_deterministic_and_sensitive(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "c.yaml").write_text("k: v\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "big.parquet").write_bytes(b"ignored")

    h1 = tree_hash(tmp_path)
    assert h1.startswith("sha256:") and h1 == tree_hash(tmp_path)

    (tmp_path / "data" / "big.parquet").write_bytes(b"changed")
    assert tree_hash(tmp_path) == h1

    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"junk")
    assert tree_hash(tmp_path) == h1

    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert tree_hash(tmp_path) != h1


def test_code_hash_without_git(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    assert code_hash(tmp_path) == tree_hash(tmp_path)
