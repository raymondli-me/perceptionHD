#!/usr/bin/env python3
"""Test that the package imports correctly."""

try:
    from perceptionhd import PerceptionHDPipeline
    print("✓ Successfully imported PerceptionHDPipeline")
    
    from perceptionhd import __version__
    print(f"✓ PerceptionHD version: {__version__}")
    
    # Test that we can instantiate the pipeline
    print("\nTesting pipeline initialization...")
    pipeline = PerceptionHDPipeline(
        data_path="examples/ai_social_class/data.csv",
        embeddings_path="examples/ai_social_class/embeddings.npy",
        config_path="examples/ai_social_class/config.yaml",
        output_dir="test_output"
    )
    print("✓ Pipeline initialized successfully")
    
    print("\nAll imports successful! Package is ready to use.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()