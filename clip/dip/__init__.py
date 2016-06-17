# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 14, 2016
#   Author: William Ro
#
########################################################################

import os

import clip.cl as cl

# region : Load kernel sources

# > locate dir kernels
path = os.path.dirname(__file__) + r'/kernels/'
# > get all kernel sources
files = os.listdir(path)
# > read sources into a string
kernel_sources = ''
for f in files:
    if f[-2:] == 'cl':
        kernel_sources += '/* ' + f + ' */\n'
        kernel_sources += open(path + f).read() + '\n'

# endregion : Load kernel sources

# region : Initialize CL

# > create OpenCL context
cl.create_context()
# > create and build program on context
cl.create_build_program(kernel_sources)

print('-=> dip.ip initialized')

# endregion : Initialize CL

# region : Self-test

if __name__ == '__main__':
    import numpy as np

    x = np.zeros((2, 2))
    buf = cl.create_buffer(cl.mem_access_mode.READ_WRITE,
                           hostbuf=x, host_ptr_mode=
                           cl.mem_host_ptr_mode.COPY_HOST_PTR)
    cl.program.test_2(x.shape, (buf,))
    # cl.program.cl_test((2, 2, 1), (np.int32(8),))


# endregion : Self-test
