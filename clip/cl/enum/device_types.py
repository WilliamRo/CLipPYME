import pyopencl as cl

ALL = cl.device_type.ALL
CPU = cl.device_type.CPU
GPU = cl.device_type.GPU
CUSTOM = cl.device_type.CUSTOM
DEFAULT = cl.device_type.DEFAULT
ACCELERATOR = cl.device_type.ACCELERATOR

del cl
