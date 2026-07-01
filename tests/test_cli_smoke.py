import pytest
from decodebench.cli import build_parser, main

def test_cli_parser_demo():
    parser = build_parser()
    args = parser.parse_args(["demo", "f1", "--dim", "2048", "--batch", "4", "--dry-run"])
    assert args.command == "demo"
    assert args.name == "f1"
    assert args.dim == 2048
    assert args.batch == 4
    assert args.dry_run is True

def test_cli_parser_profile():
    parser = build_parser()
    args = parser.parse_args(["profile", "module.py:build_fn", "--trials", "50", "--dry-run"])
    assert args.command == "profile"
    assert args.target == "module.py:build_fn"
    assert args.trials == 50
    assert args.dry_run is True

def test_cli_parser_sweep():
    parser = build_parser()
    args = parser.parse_args(["sweep", "f4", "--batch", "1,2,4", "--dim", "2048", "--dry-run"])
    assert args.command == "sweep"
    assert args.name == "f4"
    assert args.batch == [1, 2, 4]
    assert args.dim == 2048
    assert args.dry_run is True

def test_cli_dry_run_outputs(capsys):
    # Test demo dry-run
    rc = main(["demo", "f1", "--dim", "4096", "--batch", "1", "--dry-run"])
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "[dry-run] demo f1 with dim=4096, batch=1" in out

    # Test profile dry-run
    rc = main(["profile", "user_module.py:build_seq", "--dry-run"])
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "[dry-run] profile target user_module.py:build_seq" in out

    # Test sweep dry-run
    rc = main(["sweep", "f2", "--batch", "1,2", "--dim", "4096", "--dry-run"])
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "[dry-run] sweep demo f2 with batch=[1, 2], dim=4096" in out

def test_cli_profile_validation(capsys):
    # Missing colon in target
    rc = main(["profile", "user_module.py"])
    assert rc == 2
    _, err = capsys.readouterr()
    assert "Error: Target must be in the format 'path/to/module.py:build_fn'" in err

def test_cli_invalid_demo_choice():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "invalid_name"])
