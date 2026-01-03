import os
import json
import shutil
import time
from typing import Dict, Any, Optional

class ModelRegistry:
    """
    A lightweight, file-system-based model registry designed for edge environments.
    
    This class implements the Singleton pattern to ensure a shared in-memory cache 
    across the application lifecycle. It manages model artifacts and metadata 
    persistence without external database dependencies.

    Attributes:
        base_dir (str): Root directory for storing registry data and artifacts.
        _in_memory_cache (Dict): Singleton cache storage for loaded model objects.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Ensure strictly one instance per process (Singleton Pattern).
        If an instance already exists, return it; otherwise, create a new one.
        """
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_dir: str = "./data"):
        """
        Initialize the registry.
        
        Args:
            base_dir (str): Path to the storage directory. Defaults to "./data".
        """
        # Prevent re-initialization if the instance already exists
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.registry_path = os.path.join(base_dir, "registry.json")
        
        # Initialize singleton cache state
        self._in_memory_cache: Dict[str, Any] = {}
        
        # Ensure storage structure exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        if not os.path.exists(self.registry_path):
            self._save_registry({})
            
        # Mark as initialized to prevent overwriting state on subsequent calls
        self._initialized = True

    def _load_registry(self) -> Dict:
        """Internal method to load metadata from the JSON persistence layer."""
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # Fallback to empty registry in case of corruption or missing file
            return {}

    def _save_registry(self, data: Dict) -> None:
        """
        Save metadata to JSON file using an atomic write pattern.
        
        This prevents data corruption if the process crashes during a write operation.
        """
        temp_path = self.registry_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=4)
        
        # Atomic replacement
        shutil.move(temp_path, self.registry_path)

    def register_model(self, name: str, version: str, model_path: str, 
                      metrics: Dict = None, metadata: Dict = None) -> Dict:
        """
        Register a new model artifact into the system.

        Args:
            name (str): The unique name of the model family (e.g., 'YOLOv8').
            version (str): Semantic version string (e.g., 'v1.0.0').
            model_path (str): Local path to the source model file (.pt, .onnx).
            metrics (Dict, optional): Performance metrics (accuracy, latency).
            metadata (Dict, optional): Additional info (author, framework).

        Returns:
            Dict: The registered model metadata.
        """
        if metrics is None: metrics = {}
        if metadata is None: metadata = {}

        # 1. Prepare destination directory for artifact isolation
        model_dir = os.path.join(self.models_dir, name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        filename = os.path.basename(model_path)
        dest_path = os.path.join(model_dir, filename)
        
        # 2. Copy artifact (Simulating storage upload)
        shutil.copy2(model_path, dest_path)
        
        # 3. Update persistent metadata
        registry_data = self._load_registry()
        
        if name not in registry_data:
            registry_data[name] = {}
            
        model_info = {
            "name": name,
            "version": version,
            "path": os.path.abspath(dest_path),
            "metrics": metrics,
            "metadata": metadata,
            "registered_at": time.time()
        }
        
        registry_data[name][version] = model_info
        self._save_registry(registry_data)
        
        return model_info

    def get_model_path(self, name: str, version: str) -> Optional[str]:
        """Retrieve the absolute file path of a registered model."""
        data = self._load_registry()
        return data.get(name, {}).get(version, {}).get("path")

    def load_model(self, name: str, version: str) -> Any:
        """
        Load a model object with an optimized caching strategy.

        Strategy:
            1. Check In-Memory Cache (Warm Start).
            2. If missing, load from Disk (Cold Start).
            3. Update Cache for future access.

        Returns:
            Any: The loaded model object (binary content for demo).
        
        Raises:
            ValueError: If the model is not found in the registry.
        """
        # Generate unique cache key
        cache_key = f"{name}_{version}"
        
        # 1. Warm Start: Return directly from singleton cache
        if cache_key in self._in_memory_cache:
            return self._in_memory_cache[cache_key]
            
        # 2. Cold Start: Validate path and load from disk
        path = self.get_model_path(name, version)
        if not path or not os.path.exists(path):
            raise ValueError(f"Model artifact not found: {name} version {version}")
            
        # Simulating heavy model loading (e.g., torch.load)
        with open(path, "r") as f:
            model_obj = f.read() 
            
        # 3. Update Cache (Lazy Loading)
        self._in_memory_cache[cache_key] = model_obj
        
        return model_obj