"""
Python Naming Convention:
module_name, package_name, ClassName, method_name, ExceptionName,
function_name, GLOBAL_CONSTANT_NAME, global_var_name,
instance_var_name, function_parameter_name, local_var_name
"""

import clip.cl as cl
import numpy as np

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)
res_np = np.empty_like(a_np)

src = """
__kernel void sum(__global const float *a_g,
                  __global const float *b_g,
                  __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
"""

# create context
context = cl.create_context(cl.device_types.GPU)
print('>>> Context created')

# create and build program
program = context.create_build_program(src)
print('>>> Program created and built')

# create buffers
am = cl.mem_access_mode
hm = cl.mem_host_ptr_mode
a_g = context.create_buffer(am.READ_ONLY, hostbuf=a_np,
                            host_ptr_mode=hm.COPY_HOST_PTR)
b_g = context.create_buffer(am.READ_ONLY, hostbuf=b_np,
                            host_ptr_mode=hm.COPY_HOST_PTR)
res_g = context.create_buffer(am.WRITE_ONLY, a_np.nbytes)
print('>>> Buffers created')

# execute kernel
ev = program.sum(a_np.shape, (a_g, b_g, res_g))
print('>>> Kernel executed')

# read back data
res_g.enqueue_read(res_np)
print('>>> Data read back')

# Check on CPU with Numpy:
print('>>> Verifying...')
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))

# Profile
print(ev.profile_queued_end)
