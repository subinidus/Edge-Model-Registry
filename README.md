# Edge-Model-Registry

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Production--Ready-orange)

**A lightweight, zero-dependency model registry designed for isolated Edge AI environments.**

## 1. Project Background & Problem Definition

In typical MLOps pipelines, model registries rely on heavy infrastructure like SQL databases or Cloud Storage (S3). However, **Edge Devices** (e.g., Medical Devices, Factory Controllers) operate under strict constraints:
* **No Internet Connection:** Cloud-based registries are inaccessible.
* **Limited Resources:** Heavy DB processes consume vital RAM/CPU needed for inference.
* **Data Persistence:** Metadata must survive system reboots without external dependencies.

**Edge-Model-Registry** solves these problems by providing a file-system-based registry with **ACID-like persistence** and **memory optimization**.

---

## 2. Key Features

### 🔹 1. Zero-Dependency Persistence
* Uses a **JSON-based storage engine** instead of SQL/NoSQL databases.
* Ensures metadata retention across system reboots.
* **Why JSON?** For edge setups handling <1,000 models, file I/O overhead is negligible compared to the maintenance cost of a DB server.

### 🔹 2. Artifact Isolation & Versioning
* Enforces **Semantic Versioning** (e.g., `v1.0.0`, `v1.2.0`).
* Physically isolates model binaries in structured directories to prevent file conflicts.
* Directory Structure: `data/models/{model_name}/{version}/artifact.pt`

### 🔹 3. Memory Optimization (Singleton Pattern)
* Implements **Lazy Loading** with In-Memory Caching.
* Reduces redundant disk I/O and prevents memory bloat during high-frequency inference requests.
* **Benchmark:** Achieved **~17x speedup** in warm-start loading (0.12ms → 0.006ms).

---

## 3. Installation

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Edge-Model-Registry.git](https://github.com/YourUsername/Edge-Model-Registry.git)

# Install dependencies (Minimal)
pip install -r requirements.txt
```

## 4. Quick Start
You can verify the system capabilities by running the demo script:
```bash
python examples/quick_start.py
```
### Usage Example (Python API)
```bash
from edge_registry import ModelRegistry

# 1. Initialize Registry (Auto-creates storage at './data')
registry = ModelRegistry(base_dir="./data")

# 2. Register a Model
model = registry.register_model(
    name="YOLOv8-Nano",
    version="v1.0.0",
    model_path="./my_model.pt",
    metrics={"mAP": 0.85, "latency": 15.2}
)

# 3. Load Model (With Caching)
# First call: Loads from disk
model_obj = registry.load_model("YOLOv8-Nano", "v1.0.0")

# Second call: Returns cached object instantly
model_obj_cached = registry.load_model("YOLOv8-Nano", "v1.0.0")
```

## 5. System Architecture

The system is designed with a 3-Tier Architecture to ensure separation of concerns.

| Layer | Component | Responsibility |
| :--- | :--- | :--- |
| **Interface** | `core.py` | API Surface, Validation, Exception Handling |
| **Caching** | `_in_memory_cache` | Singleton Management, Lazy Loading |
| **Storage** | `registry.json` | Metadata Persistence, File System Mapping |

## 6. Performance Benchmark

Tested on standard environment (Intel i7 / Windows 11):

| Operation | Access Type | Latency (ms) | Note |
| :--- | :--- | :--- | :--- |
| **Cold Start** | Disk I/O | 0.1216 ms | First time access |
| **Warm Start** | Memory Cache | **0.0069 ms** | **17.6x Faster** |

## 7. Future Work (Roadmap)

* [ ] Add **File Locking** mechanism for multi-process safety.
* [ ] Support **S3 Integration** for hybrid (Cloud-Edge) sync.
* [ ] Add CLI support (e.g., `registry list`, `registry push`).