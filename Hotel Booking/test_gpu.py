import torch
import time

def test_gpu():
    print("--- GPU Hardware Test ---")
    
    # 1. Check if CUDA (NVIDIA GPU support) is available
    if not torch.cuda.is_available():
        print("Result: GPU not detected. Ensure drivers and CUDA Toolkit are installed.")
        return

    # 2. Identify the Device
    gpu_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda")
    print(f"Device Found: {gpu_name}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    # 3. Performance Test: Large Matrix Multiplication
    # We create two 10,000 x 10,000 matrices
    size = 10000
    print(f"\nInitializing {size}x{size} matrices on GPU...")
    
    try:
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)

        print("Running matrix multiplication stress test...")
        start_time = time.time()
        
        # Perform the operation
        result = torch.matmul(x, y)
        
        # CUDA operations are asynchronous, so we need to synchronize to get accurate timing
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        print(f"Test Successful!")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        
    except RuntimeError as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_gpu()