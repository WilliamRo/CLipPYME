from pyopencl import command_queue_properties as p

OUT_OF_ORDER_EXEC_MODE_ENABLE = p.OUT_OF_ORDER_EXEC_MODE_ENABLE
PROFILING_ENABLE = p.PROFILING_ENABLE
# Available with OpenCL 2.0
if 'ON_DEVICE' in dir(p):
    ON_DEVICE = p.ON_DEVICE
if '' in dir(p):
    ON_DEVICE_DEFAULT = p.ON_DEVICE_DEFAULT

del p
