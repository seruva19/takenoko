# setup.py

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Determine GPU architecture automatically
def get_gpu_arch_flags():
    """Get appropriate GPU architecture flags for current system."""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using default compute capability")
        return ['-gencode=arch=compute_80,code=sm_80']
    
    # Get GPU compute capabilities
    gpu_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        arch = major * 10 + minor
        gpu_archs.append(arch)
    
    # Generate architecture flags
    arch_flags = []
    unique_archs = sorted(set(gpu_archs))
    
    for arch in unique_archs:
        arch_flags.append(f'-gencode=arch=compute_{arch},code=sm_{arch}')
    
    print(f"🔧 Detected GPU architectures: {unique_archs}")
    print(f"🔧 Using compute capabilities: {arch_flags}")
    
    return arch_flags

setup(
    name='stochastic_ops_cuda',
    ext_modules=[
        CUDAExtension(
            name='stochastic_ops_cuda',
            sources=['stochastic_ops.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': get_gpu_arch_flags() + [
                    '-lineinfo',
                    '--use_fast_math',  # Enable fast math for better performance
                ],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
