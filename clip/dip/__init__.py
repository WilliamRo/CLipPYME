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

# region : Self-test

if __name__ == '__main__':
    context = cl.create_context()
    print context.device_list[0].name


# endregion : Self-test
