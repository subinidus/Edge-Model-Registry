import os
import sys
import time
import shutil

# [Setup] Add 'src' to python path to allow direct import of the library
# This enables running the script directly from the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from edge_registry.core import ModelRegistry

def main():
    print("\n" + "="*60)
    print("Edge-Model-Registry | System Capability Demo")
    print("="*60)
    
    # Configuration for the demo
    DEMO_DIR = "./demo_storage"
    MODEL_NAME = "MobileNetV2_Quantized"
    VERSION = "v1.2.0"
    
    # 1. System Initialization
    print(f"\n[INFO] Initializing Registry System at '{DEMO_DIR}'...")
    registry = ModelRegistry(base_dir=DEMO_DIR)

    # ---------------------------------------------------------
    # Scenario 1: Model Registration (Write Operation)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" [Scenario 1] Registering New Model Artifact")
    print("-"*-40)
    
    # Generate a dummy model artifact for demonstration
    dummy_source = "temp_model_artifact.pt"
    with open(dummy_source, "w") as f:
        f.write("Binary content of the model...")

    try:
        # Register the model
        start_t = time.time()
        model_info = registry.register_model(
            name=MODEL_NAME,
            version=VERSION,
            model_path=dummy_source,
            metrics={"accuracy": 0.925, "inference_time_ms": 4.5},
            metadata={"author": "Subin Seo", "target_device": "Jetson Nano"}
        )
        elapsed = (time.time() - start_t) * 1000
        
        print(f" [SUCCESS] Model Registered Successfully.")
        print(f"   ├── Name:    {model_info['name']}")
        print(f"   ├── Version: {model_info['version']}")
        print(f"   ├── Path:    {model_info['path']}")
        print(f"   └── Time:    {elapsed:.2f} ms")

    except ValueError as e:
        print(f" [WARNING] Registration Skipped: {e}")
    except Exception as e:
        print(f" [ERROR] Unexpected Error: {e}")
    finally:
        # Clean up source artifact (simulate move/upload)
        if os.path.exists(dummy_source):
            os.remove(dummy_source)

    # ---------------------------------------------------------
    # Scenario 2: Data Persistence (Recovery Operation)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" [Scenario 2] Verifying Data Persistence (Simulate Restart)")
    print("-"*-40)
    
    print(" [INFO] Re-initializing Registry Instance...")
    # Create a new instance to simulate a fresh process/server restart
    new_registry = ModelRegistry(base_dir=DEMO_DIR)
    
    saved_path = new_registry.get_model_path(MODEL_NAME, VERSION)
    
    if saved_path:
        print(f" [CHECK] Integrity Verified: Metadata successfully loaded from disk.")
        print(f" [CHECK] Target Artifact Exists: {os.path.exists(saved_path)}")
    else:
        print(f" [FAIL] Critical: Metadata lost after restart.")

    # ---------------------------------------------------------
    # Scenario 3: Caching Mechanism (Performance Optimization)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print(" [Scenario 3] Benchmarking Lazy Loading & Caching")
    print("-"*-40)
    
    # First Load: Simulating Disk I/O (Cold Start)
    t0 = time.time()
    model_obj_1 = new_registry.load_model(MODEL_NAME, VERSION)
    t1 = time.time()
    cold_start_time = (t1 - t0) * 1000
    
    print(f" [1st Load] Cold Start (Disk Access) : {cold_start_time:.4f} ms")

    # Second Load: Simulating Cache Hit (Warm Start)
    t2 = time.time()
    model_obj_2 = new_registry.load_model(MODEL_NAME, VERSION)
    t3 = time.time()
    warm_start_time = (t3 - t2) * 1000
    
    print(f" [2nd Load] Warm Start (Cache Hit)   : {warm_start_time:.4f} ms")
    
    # Verification
    is_singleton = (model_obj_1 is model_obj_2)
    speedup = cold_start_time / (warm_start_time + 1e-9) # Prevent div by zero
    
    print(f" [RESULT] Singleton Object: {is_singleton}")
    if is_singleton:
        print(f" [RESULT] Performance Gain: {speedup:.1f}x Faster")

    print("\n" + "="*60)
    print("Demo Completed Successfully.")
    print("="*60 + "\n")

    # Optional: Clean up demo storage for repeated tests
    # shutil.rmtree(DEMO_DIR) 

if __name__ == "__main__":
    main()