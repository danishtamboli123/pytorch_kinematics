"""
Memory management utilities for PyTorch Kinematics to prevent memory leaks
"""
import gc
import torch
from contextlib import contextmanager


def clear_gpu_memory():
    """Force clear GPU memory and garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "CUDA not available"


@contextmanager
def memory_managed_context():
    """Context manager that ensures memory cleanup"""
    try:
        yield
    finally:
        clear_gpu_memory()


def detach_tensors_recursive(obj):
    """Recursively detach tensors from computational graph"""
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(detach_tensors_recursive(item) for item in obj)
    else:
        return obj


class MemoryTracker:
    """Track memory usage throughout training"""
    
    def __init__(self):
        self.start_memory = None
        self.peak_memory = 0
        
    def start(self):
        clear_gpu_memory()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        
    def update(self):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated()
            self.peak_memory = max(self.peak_memory, current)
            
    def report(self):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
            start = self.start_memory or 0
            
            print(f"Memory Usage Report:")
            print(f"  Start: {start / 1024**3:.2f}GB")
            print(f"  Current: {current / 1024**3:.2f}GB") 
            print(f"  Peak: {peak / 1024**3:.2f}GB")
            print(f"  Increase: {(current - start) / 1024**3:.2f}GB")
