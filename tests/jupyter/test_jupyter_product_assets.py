from pathlib import Path


def test_welcome_and_assets_exist():
    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "jupyter" / "notebooks" / "welcome.ipynb").exists()
    assert (repo_root / "jupyter" / "notebooks" / "new_project_template.ipynb").exists()
    assert (repo_root / "jupyter" / "data" / "iris.csv").exists()
    assert (repo_root / "jupyter" / "settings" / "@jupyterlab" / "launcher-extension" / "launcher.jupyterlab-settings").exists()


def test_jupyter_config_opens_welcome():
    repo_root = Path(__file__).resolve().parents[2]
    cfg = (repo_root / "jupyter" / "jupyter_config.py").read_text()
    assert "welcome.ipynb" in cfg
