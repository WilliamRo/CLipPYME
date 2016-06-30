# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl


class MemoryObject(cl.MemoryObject):
    """OpenCL MemoryObject

    Derived from pyopencl.MemoryObject

    """

    # region : Constructor

    def __init__(self, hostbuf=None):
        super(MemoryObject, self).__init__(hostbuf)

    # endregion

    # region : Properties

    @property
    def associated_memobject(self):
        return super(MemoryObject, self).associated_memobject

    @property
    def context(self):
        from ..cl import context
        return context

    @property
    def flags(self):
        return super(MemoryObject, self).flags

    @property
    def host_ptr(self):
        return super(MemoryObject, self).host_ptr

    @property
    def map_count(self):
        return super(MemoryObject, self).map_count

    @property
    def offset(self):
        return super(MemoryObject, self).offset

    @property
    def size(self):
        return super(MemoryObject, self).size

    @property
    def type(self):
        return super(MemoryObject, self).type

    @property
    def reference_count(self):
        return super(MemoryObject, self).reference_count

    # endregion

    # region : Public Methods

    def enqueue_read(self, dest, queue=None,
                     is_blocking=True, wait_for=[]):
        # if queue is None, set default queue
        if queue is None:
            queue = self.context.default_queue
        # read memory to dest
        return cl.enqueue_copy(queue, dest, self,
                               is_blocking=is_blocking,
                               wait_for=wait_for)

    def enqueue_write(self, src, queue=None,
                     is_blocking=True, wait_for=[]):
        # if queue is None, set default queue
        if queue is None:
            queue = self.context.default_queue
        # write memory to device
        return cl.enqueue_copy(queue, self, src,
                               is_blocking=is_blocking,
                               wait_for=wait_for)

    # endregion : Public Methods

    pass  # patch for region
