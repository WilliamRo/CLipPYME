# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 1, 2016
#   Author: William Ro
#
########################################################################

from .enum import queue_properties

import pyopencl as cl


class Device(cl.Device):
    """OpenCL Device

    Derived from pyopencl.Device

    Properties: name, vendor, version, max_compute_units, max_work_item_sizes
                max_mem_alloc_size, global_mem_size, image_support ...

    """

    # region : Constructor

    def __init__(self):
        pass

    # endregion

    # region : Properties

    @property
    def extensions_list(self):
        return self.extensions.split(' ')

    @property
    def type_str(self):
        index = self.type
        if index == cl.device_type.CPU:
            return 'CPU'
        elif index == cl.device_type.GPU:
            return 'GPU'
        elif index == cl.device_type.ACCELERATOR:
            return 'Accelerator'
        else:
            return 'Default Type'

    # endregion

    # region : Public Methods

    def info(self, show_ext=False):
        info = ''
        info += 'Device Name:         ' + self.name + '\n'
        info += 'Vendor:              ' + self.vendor + '\n'
        info += 'Type:                ' + self.type_str + '\n'
        info += 'Version:             ' + self.version + '\n'
        info += 'Max Compute Units:   ' + str(self.max_compute_units) + '\n'
        info += 'Max Work Item Sizes: ' + str(self.max_work_item_sizes) + '\n'
        info += 'Max Work Group Size: ' + str(self.max_work_group_size) + '\n'
        info += 'Max Mem Alloc Size:  ' + str(self.max_mem_alloc_size / 1024 / 1024) + ' MB\n'
        info += 'Global Memory Size:  ' + str(self.global_mem_size / 1024 / 1024) + ' MB\n'
        info += 'Image Support:       ' + str(self.image_support) + '\n'

        if show_ext:
            info += 'Extensions:          ' + '\n\t\t\t\b\b\b'.join(self.extensions_list) + '\n'

        return info

    def create_queue(self, properties=queue_properties.PROFILING_ENABLE):
        if 'context' not in dir(self):
            raise Exception('!!! Failed to create queue(context not found)')
        # import
        from .command_queue import CommandQueue
        # create command queue
        queue = CommandQueue(self.context, self, properties)
        # put queue in list
        self.queues.append(queue)

        return queue

    # endregion

    pass  # patch for region
