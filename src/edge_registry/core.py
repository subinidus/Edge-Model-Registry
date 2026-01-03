import os
import json
import shutil
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

class ModelRegistry:
    """
    A file-system based lightweight model registry for edge environments.
    
    This class handles the lifecycle of ML models including registration,
    versioning, storage isolation, metadata persistence, and in-memory caching.
    """

    def __init__(self, base_dir: str = "./data"):
        """
        Initialize the ModelRegistry system.

        Args:
            base_dir (str): The root directory where models and metadata will be stored.
                            Defaults to "./data".
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.registry_path = os.path.join(base_dir, "registry.json")
        
        # [Optimization] In-memory cache to prevent redundant disk I/O and memory usage
        # Key: "{name}_{version}", Value: Loaded Model Object
        self._in_memory_cache: Dict[str, Any] = {}
        
        # Ensure the storage directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize persistence layer
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """
        Load metadata from the JSON registry file.
        """
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[Warning] Registry file at {self.registry_path} is corrupted. Initializing new registry.")
                return {"models": []}
        return {"models": []}

    def _save_registry(self):
        """
        Persist current metadata to the JSON registry file.
        """
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=4, ensure_ascii=False)

    def register_model(self, name: str, version: str, model_path: str, 
                       metrics: Dict[str, float], metadata: Dict[str, str] = None) -> Dict:
        """
        Register a new model version and isolate the artifact.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source model artifact not found: {model_path}")

        if self.get_model_path(name, version):
            raise ValueError(f"Model '{name}' version '{version}' already exists in registry.")

        target_dir = os.path.join(self.models_dir, name, version)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = os.path.basename(model_path)
        target_path = os.path.join(target_dir, filename)
        
        shutil.copy2(model_path, target_path)

        model_info = {
            "name": name,
            "version": version,
            "path": os.path.abspath(target_path), 
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        }
        
        self.registry["models"].append(model_info)
        self._save_registry()
        
        return model_info

    def get_model_path(self, name: str, version: str) -> Optional[str]:
        """Retrieve the absolute file path for a specific model version."""
        for model in self.registry["models"]:
            if model["name"] == name and model["version"] == version:
                return model["path"]
        return None

    def load_model(self, name: str, version: str) -> Any:
        """
        Load a model into memory with Caching (Singleton-like pattern).

        If the model is already in the cache, return the cached instance.
        Otherwise, load it from disk, cache it, and return it.

        Args:
            name (str): The model name.
            version (str): The target version.

        Returns:
            Any: The loaded model object (e.g., PyTorch Module).
        
        Raises:
            ValueError: If the model is not found in the registry.
        """
        # 1. Check Cache (Hit)
        cache_key = f"{name}_{version}"
        if cache_key in self._in_memory_cache:
            # print(f"[System] Serving '{cache_key}' from In-Memory Cache.") # For debugging
            return self._in_memory_cache[cache_key]

        # 2. Check Disk (Miss)
        path = self.get_model_path(name, version)
        if not path:
            raise ValueError(f"Model '{name}' version '{version}' not found in registry.")

        # 3. Simulate Heavy Loading (In real usage: model = torch.load(path))
        # Here we assume it's a generic object or simulate loading for the portfolio demo.
        print(f"[System] Loading '{cache_key}' from Disk... (Simulated Heavy I/O)")
        
        # [PORTFOLIO NOTE]: Replace this logic with actual `torch.load(path)` if needed.
        # For now, we wrap the path in a dummy class to simulate an object.
        loaded_instance = f"Loaded Model Object ({name} {version}) from {path}"
        
        # 4. Update Cache
        self._in_memory_cache[cache_key] = loaded_instance
        
        return loaded_instance