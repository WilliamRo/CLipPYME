# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 1, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl

from device import Device


class Platform(cl.Platform):
    """OpenCL Platform

    Derived from pyopencl.Platform

    Properties: name, vendor, version, profile, extensions
    """

    # region : Constructor

    def __init__(self):
        self.devices = []

    # endregion

    # region : Properties

    @property
    def extensions_list(self):
        return self.extensions.split(' ')

    # endregion

    # region : Public Methods

    def info(self, show_ext=False):
        info = ''
        info += '       Name: ' + self.name + '\n'
        info += '     Vendor: ' + self.vendor + '\n'
        info += ' CL Version: ' + self.version + '\n'
        info += '    Profile: ' + self.profile + '\n'
        if show_ext:
            info += ' Extensions: ' + '\n\t\t\b\b\b'.join(self.extensions_list) + '\n'

        return info

    def details(self, show_ext=False):
        details = ''
        details += self.info() + '\n'
        for i in range(len(self.devices)):
            details += '[Device %d]\n' % i
            details += self.devices[i].info(show_ext) + '\n'

        return details

    # endregion

    # region : Description

    @staticmethod
    def get_platforms():
        """Get all CL platforms
        """
        # Get all CL platforms
        platforms = cl.get_platforms()
        for p in platforms:
            # Not a safe way
            p.__class__ = Platform
            p.devices = p.get_devices()
            for d in p.devices:
                d.__class__ = Device
        return platforms

    pass  # patch for region
    # endregion
