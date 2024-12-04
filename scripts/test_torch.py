import torch

def test_torch_cuda():
    # Test PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")
    
    if cuda_available:
        # Print CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA device name: {device_name}")
        
        # Test CUDA tensor creation
        print("\nTesting CUDA tensor creation:")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"Test tensor on CUDA: {x}")
        print(f"Tensor device: {x.device}")
    else:
        print("CUDA is not available. PyTorch will run on CPU only.")
        
    # Test CPU tensor creation
    print("\nTesting CPU tensor creation:")
    y = torch.tensor([4.0, 5.0, 6.0])
    print(f"Test tensor on CPU: {y}")
    print(f"Tensor device: {y.device}")

if __name__ == "__main__":
    test_torch_cuda()
