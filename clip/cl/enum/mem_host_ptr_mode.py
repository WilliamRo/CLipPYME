from pyopencl import mem_flags as mf

USE_HOST_PTR = mf.USE_HOST_PTR
ALLOC_HOST_PTR = mf.ALLOC_HOST_PTR
COPY_HOST_PTR = mf.COPY_HOST_PTR
ALLOC_COPY_HOST_PTR = ALLOC_HOST_PTR | COPY_HOST_PTR
DEFAULT = 0

# Available with OpenCL 2.0
KERNEL_READ_AND_WRITE = mf.KERNEL_READ_AND_WRITE

# Available with the cl_amd_device_memory_flags extension
USE_PERSISTENT_MEM_AMD = mf.USE_PERSISTENT_MEM_AMD

del mf
