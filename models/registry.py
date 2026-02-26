"""Model registry — save, load, and compare trained models."""

import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb

MODELS_DIR = Path(__file__).parent / "saved"


class ModelRegistry:
    """Manages versioned model storage."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: lgb.Booster, metadata: dict | None = None) -> str:
        """Save a model with timestamp-based versioning. Returns version string."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.base_dir / f"model_{version}.txt"
        meta_path = self.base_dir / f"model_{version}_meta.json"

        model.save_model(str(model_path))

        meta = metadata or {}
        meta["version"] = version
        meta["model_path"] = str(model_path)
        meta["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update "latest" symlink-style pointer
        latest_path = self.base_dir / "latest_version.txt"
        latest_path.write_text(version)

        return version

    def load(self, version: str | None = None) -> lgb.Booster:
        """Load a model by version. If None, loads latest."""
        if version is None:
            version = self._latest_version()
        model_path = self.base_dir / f"model_{version}.txt"
        return lgb.Booster(model_file=str(model_path))

    def load_metadata(self, version: str | None = None) -> dict:
        """Load metadata for a model version."""
        if version is None:
            version = self._latest_version()
        meta_path = self.base_dir / f"model_{version}_meta.json"
        with open(meta_path) as f:
            return json.load(f)

    def list_versions(self) -> list[str]:
        """List all saved model versions (newest first)."""
        versions = []
        for p in self.base_dir.glob("model_*_meta.json"):
            v = p.stem.replace("model_", "").replace("_meta", "")
            versions.append(v)
        return sorted(versions, reverse=True)

    def _latest_version(self) -> str:
        latest_path = self.base_dir / "latest_version.txt"
        if latest_path.exists():
            return latest_path.read_text().strip()
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError("No saved models found")
        return versions[0]
