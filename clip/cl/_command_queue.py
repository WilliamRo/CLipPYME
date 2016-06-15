# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl


class CommandQueue(cl.CommandQueue):
    """OpenCL MemoryObject

    Derived from pyopencl.CommandQueue

    """

    # region : Constructor

    def __init__(self, context, device=None, properties=None):
        super(CommandQueue, self).__init__(context, device, properties)

    # endregion

    # region : Properties

    @property
    def context(self):
        from ..cl import context
        return context

    @property
    def device(self):
        return super(CommandQueue, self).device

    @property
    def properties(self):
        return super(CommandQueue, self).properties

    @property
    def reference_count(self):
        return super(CommandQueue, self).reference_count

    # endregion

    pass  # patch for region
