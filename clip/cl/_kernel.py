# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl


class Kernel(cl.Kernel):
    """OpenCL MemoryObject

    Derived from pyopencl.Kernel

    """

    # region : Constructor

    def __init__(self, program, name):
        super(Kernel, self).__init__(program, name)

    # endregion : Constructor

    # region : Properties

    @property
    def attributes(self):
        return super(Kernel, self).attributes

    @property
    def context(self):
        from ..cl import context
        return context

    @property
    def function_name(self):
        return super(Kernel, self).function_name

    @property
    def num_agrs(self):
        return super(Kernel, self).num_agrs

    @property
    def program(self):
        return super(Kernel, self).program

    @property
    def reference_count(self):
        return super(Kernel, self).reference_count

    # endregion : Properties

    # region : Operator Overloading

    def __call__(self, global_size=(1,),
                 args=(), local_size=None,
                 queue=None, global_offset=None,
                 wait_for=None, g_times_l=False):
        # import
        from ._event import Event
        # if queue is None, set default queue
        if queue is None:
            queue = self.context.default_queue
        # call pyopencl.Kernel.__call__
        evt = super(Kernel, self).__call__(
            queue, global_size, local_size, *args,
            global_offset=global_offset,
            wait_for=wait_for,
            g_times_l=g_times_l)
        evt.__class__ = Event
        return evt

    # endregion : Operator Overloading

    # region : Public Methods

    def enqueue_nd_range(self, global_size, queue=None,
                         local_size=None, global_offset=None,
                         wait_for=None, g_times_l=False):
        # if queue is None, set default queue
        if queue is None:
            queue = self.context.default_queue
        # call pyopencl.enqueue_nd_range_kernel
        return cl.enqueue_nd_range_kernel(
            queue, self, global_size, local_size,
            global_work_offset=global_offset,
            wait_for=wait_for, g_times_l=g_times_l)

    # endregion : Public Methods

    pass  # patch for region
