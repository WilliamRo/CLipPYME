# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl

from _kernel import Kernel


class Program(cl.Program):
    """OpenCL MemoryObject

    Derived from pyopencl.Program

    """

    # region : Constructor

    def __init__(self, arg1, arg2=None, arg3=None):
        super(Program, self).__init__(arg1, arg2, arg3)

    # endregion : Constructor

    # region : Properties

    @property
    def binaries(self):
        return super(Program, self).binaries

    @property
    def binary_sizes(self):
        return super(Program, self).binary_sizes

    @property
    def context(self):
        from ..cl import context
        return context

    @property
    def devices(self):
        return super(Program, self).devices

    @property
    def kernel_names(self):
        return super(Program, self).kernel_names

    @property
    def num_devices(self):
        return super(Program, self).num_devices

    @property
    def num_kernels(self):
        return super(Program, self).num_kernels

    @property
    def source(self):
        return super(Program, self).source

    @property
    def reference_count(self):
        return super(Program, self).reference_count

    # endregion

    # region : Operator Overloading

    def __getattr__(self, attr):
        att = super(Program, self).__getattr__(attr)
        # upgrade Kernel instance
        if isinstance(att, cl.Kernel):
            att.__class__ = Kernel
        return att

    # endregion : Operator Overloading

    pass  # patch for region
