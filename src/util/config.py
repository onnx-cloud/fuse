import json
from pathlib import Path

class ConfigError(Exception):
    pass

class ConfigLoader:
    def __init__(self, schema_dir: Path = None):
        if schema_dir is None:
            # point to the schemas folder by default
            self.schema_dir = Path(__file__).resolve().parents[2] / "schemas"
        else:
            self.schema_dir = Path(schema_dir)
        self._cache = {}
    
    def load(self, name: str, default=None, force_reload: bool = False) -> dict:
        """Load schema or config file, with caching and error handling."""
        if not force_reload and name in self._cache:
            return self._cache[name]
        
        # If absolute path or relative path to a specific file
        name_path = Path(name)
        if name_path.is_absolute() or name_path.exists():
            path = name_path
        else:
            path = self.schema_dir / f"{name}.json"
            if not path.exists():
                path = self.schema_dir / name
            
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._cache[name] = data
            return data
        except FileNotFoundError:
            if default is not None:
                return default
            raise ConfigError(f"Config or Schema not found: {name}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse JSON for {name}: {e}")

# Default global instance
_default_loader = ConfigLoader()

def load_schema(name: str, default=None, force_reload: bool = False) -> dict:
    return _default_loader.load(name, default, force_reload=force_reload)
