from pathlib import Path

THIS_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = THIS_DIR.parent
ROOT_DIR: Path = SRC_DIR.parent
DATA_DIR: Path = ROOT_DIR / "data"

