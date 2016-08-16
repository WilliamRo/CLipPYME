# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 3, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl


class Event(cl.Event):
    """OpenCL MemoryObject

    Derived from pyopencl.Event

    """

    # region : Constructor

    def __init__(self):
        super(Event, self).__init__()

    # endregion : Constructor

    # region : Properties

    # region : Properties : Info

    @property
    def command_execution_status(self):
        return super(Event, self).command_execution_status

    @property
    def command_queue(self):
        return super(Event, self).command_queue

    @property
    def command_type(self):
        return super(Event, self).command_type

    @property
    def context(self):
        from ..cl import context
        return context

    @property
    def reference_count(self):
        return super(Event, self).reference_count

    # endregion : Properties : Info

    # region : Properties : Profile Info

    @property
    def profile_complete(self):
        return super(Event, self).profile.complete

    @property
    def profile_queued(self):
        return super(Event, self).profile.queued

    @property
    def profile_submit(self):
        return super(Event, self).profile.submit

    @property
    def profile_start(self):
        return super(Event, self).profile.start

    @property
    def profile_end(self):
        return super(Event, self).profile.end

    @property
    def profile_queued_end(self):
        span = self.profile_end - self.profile_queued
        return nano_to_str(span)

    @property
    def profile_details(self):
        span1 = self.profile_submit - self.profile_queued
        span2 = self.profile_start - self.profile_submit
        span3 = self.profile_end - self.profile_start
        span4 = self.profile_end - self.profile_queued

        details = '\n'
        details += '    Queued -> Submit: ' + nano_to_str(span1) + '\n'
        details += '    Submit -> Started: ' + nano_to_str(span2) + '\n'
        details += '   Started -> Ended:   ' + nano_to_str(span3) + '\n'
        details += '              Total:   ' + nano_to_str(span4) + '\n'

        return details

    # endregion : Properties : Profile Info

    # endregion : Properties

    pass  # patch for region


def nano_to_str(span):
    """Convert nanoseconds to time string"""
    # get second, millisecond and microsecond
    sec = int(span / 1e9)
    span -= sec * sec
    ms = int(span / 1e6)
    span -= ms * 1e6
    us = int(span / 1e3)
    # generate time string
    res = ''
    if sec != 0:
        res += '%d sec ' % sec
    if ms != 0:
        res += '%d ms ' % ms
    if us != 0:
        res += '%d us' % us
    if res == '':
        res = '0'
    return res
