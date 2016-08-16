# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 1, 2016
#   Author: William Ro
#
########################################################################

"""OpenCL utilities

Modules in this package is based on Andreas Kloeckner's pyopencl:
https://documen.tician.de/pyopencl/

"""

import numpy as np

from pyopencl import enqueue_copy
from pyopencl import wait_for_events
from pyopencl import LocalMemory

from enum import device_types, vendors, \
    mem_host_ptr_mode, mem_access_mode

from _platform import Platform
from _device import Device
from _context import Context
from _command_queue import CommandQueue
from _event import Event
from _memory_object import MemoryObject

__all__ = [
    # PyOpenCL methods
    'enqueue_copy', 'wait_for_events',
    # modules
    'Platform', 'Device', 'Context',
    'CommandQueue', 'Event', 'MemoryObject',
    # variables
    'context', 'platforms',
    # enumerations
    'device_types', 'vendors'
]

# Initialize variables
real = np.float32

am = mem_access_mode
hm = mem_host_ptr_mode
context = None
CU_count = -1
program = None

# Get all CL platforms
platforms = Platform.get_platforms()


# region : Methods

def print_details(show_ext=False):
    details = ''
    for i in range(len(platforms)):
        details += '[ Platform %d ]\n' % i
        details += platforms[i].details(show_ext)

    print details


def create_context(device_type=device_types.ALL,
                   vendor=vendors.ALL,
                   device_list=None):
    """Create context on specified device list.

        If device list is not specified, context will be created
          on one device with largest global memory according
          to the given device type and vendor. A GPU device is
          preferred.
    """
    global context, CU_count
    # > if device list is specified
    if device_list is not None:
        context = Context(device_list)
        return context
    # > other wise create context on one device
    chosen_device = None
    # > scan all CL platforms
    for p in platforms:
        # > scan all devices on this platform
        for d in p.devices:
            # > check device type
            if device_type != device_types.ALL \
                    and d.type != device_type:
                continue
            # > check vendor
            if vendor != vendors.ALL and d.vendor_id != vendor:
                continue
            # > if type and vendor matches, compare with currently
            #   selected device
            if chosen_device is None \
                    or chosen_device.type != device_types.GPU \
                            and d.type == device_types.GPU:
                chosen_device = d
            elif d.global_mem_size > chosen_device.global_mem_size \
                    and d.type == device_types.GPU:
                chosen_device = d
    # if device is not found, raise exception
    if chosen_device is None:
        raise AttributeError('Can not find the specified device')
    # create context and return
    context = Context([chosen_device])
    # set CU count
    CU_count = chosen_device.max_compute_units

    return context


def create_build_program(src, devices=None,
                         options=[], cache_dir=None):
    """Create and build program on cl.context"""
    if context is None:
        raise StandardError('!!! cl.context is None')
    return context.create_build_program(src, devices,
                                        options, cache_dir)


def compile_link_program(header_infos, source_infos,
                         build_options=[],
                         compile_options=[], link_options=[],
                         info_is_filename=True):
    """Separate compilation and linking"""
    if context is None:
        raise StandardError('!!! cl.context is None')
    return context.compile_link_program(header_infos,
                                        source_infos,
                                        build_options,
                                        compile_options,
                                        link_options,
                                        info_is_filename)


def create_buffer(access_mode, size=0, hostbuf=None,
                  host_ptr_mode=mem_host_ptr_mode.DEFAULT):
    """Create buffer on cl.context"""
    return context.create_buffer(access_mode, size, hostbuf,
                                 host_ptr_mode)


def flush_default_queue():
    context.default_queue.flush()


def finish_default_queue():
    context.default_queue.finish()


# endregion : Methods


pass  # patch for region
