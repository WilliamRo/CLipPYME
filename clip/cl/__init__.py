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

from pyopencl import enqueue_copy

from enum import device_types, vendors, \
    mem_host_ptr_mode, mem_access_mode

from platform import Platform
from device import Device
from context import Context
from command_queue import CommandQueue
from event import Event
from memory_object import MemoryObject

__all__ = [
    # PyOpenCL methods
    'enqueue_copy',
    # modules
    'Platform', 'Device', 'Context',
    'CommandQueue', 'Event', 'MemoryObject',
    # variables
    'context', 'platforms',
    # enumerations
    'device_types', 'vendors'
]

# Initialize variables
context = None

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
                   vendor=vendors.ALL):
    global context
    # initialize device list
    device_list = []
    # scan all CL platforms
    for p in platforms:
        # scan all devices on this platform
        for d in p.devices:
            # check device type
            if device_type != device_types.ALL and d.type != device_type:
                continue
            # check vendor
            if vendor != vendors.ALL and d.vendor_id != vendor:
                continue
            # if type and vendor matches, break
            device_list.append(d)
            break
        # if device is found
        if len(device_list) > 0:
            break
    # if device list is empty, raise exception
    if len(device_list) == 0:
        raise AttributeError('Can not find the specified device')
    # create context and return
    context = Context(device_list)

    return context

# endregion : Methods
