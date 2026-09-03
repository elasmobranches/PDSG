"""CLI behaviour that is not about any one command's output.

The commands themselves are exercised where their subject lives -- sweep in
test_sweep.py, the seed rule in test_select_seed.py. What is here is how the
CLI behaves when its preconditions are not met.
"""


def test_smoke_from_a_directory_without_tests_says_what_to_do(tmp_path, capsys,
                                                              monkeypatch):
    """`chamnet smoke` needs a checkout; without one it must say so.

    The test files are not installed with the package, so running this command
    from anywhere else -- an installed wheel, or the container image without
    the documented volume mount -- cannot work. It used to hand pytest a bare
    relative path and let it report `file or directory not found:
    tests/test_smoke.py`, which does not tell a caller that a checkout is what
    is missing or how to supply one.
    """
    monkeypatch.chdir(tmp_path)
    # Also hide the source tree the package itself lives in, so the fallback
    # cannot rescue this the way it does for an editable install.
    import chamnet.cli as cli
    monkeypatch.setattr(cli, '__file__', str(tmp_path / 'pkg' / 'cli.py'))

    assert cli.main(['smoke']) == 2
    err = capsys.readouterr().err
    assert 'checkout' in err
    assert 'docker run' in err
    assert str(tmp_path) in err
