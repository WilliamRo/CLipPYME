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

paths = []
# > locate dir kernels
# >> dip
path = os.path.dirname(__file__) + r'/kernels/'
paths += [path]
# >> op
path = os.path.dirname(__file__)
path = os.path.dirname(path) + r'/op/kernels/'
paths += [path]

# > get all kernel sources
kernel_sources = []
kernel_headers = []

for path in paths:
    files = os.listdir(path)
    for f in files:
        if f[-2:] == 'cl':
            kernel_sources += (path + f,)
        elif f[-3:] == 'clh':
            header_name = f.split('.')[0] + '.clh'
            kernel_headers += ((path + f, header_name),)

# endregion : Load kernel sources

# region : Initialize CL

# > create OpenCL context
cl.create_context()
# > create and build program on context
cl.compile_link_program(kernel_headers, kernel_sources)

print('-=> dip.ip initialized')

# endregion : Initialize CL

# region : Self-test

if __name__ == '__main__':
    import numpy as np

    x = np.zeros((2, 2), np.int32)
    x[0, 0] = 800
    x[0, 1] = 801
    x[1, 0] = 810
    x[1, 1] = 811
    print x.flatten()
    print x.size
    x_buf = cl.create_buffer(cl.am.READ_ONLY, x.nbytes)
    x_buf.enqueue_write(x)
    cl.program.test(4, x_buf)




# endregion : Self-test
