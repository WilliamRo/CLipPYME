# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

from _memory_object import MemoryObject

import pyopencl as cl


class Buffer(cl.Buffer, MemoryObject):
    """OpenCL Buffer

    Derived from pyopencl.Buffer

    """

    # region : Constructor

    def __init__(self, context, flags, size=0, hostbuf=None):
        super(Buffer, self).__init__(context, flags, size, hostbuf)

    # endregion : Constructor

    pass  # patch for region
