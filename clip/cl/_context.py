# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 1, 2016
#   Author: William Ro
#
########################################################################

import pyopencl as cl

from enum import mem_host_ptr_mode


class Context(cl.Context):
    """OpenCL Context

    Derived from pyopencl.Context
    Context can be created on one or more devices

    Properties: devices, num_devices, properties, reference_count

    """

    # region : Constructor

    def __init__(self, devices=None, properties=None,
                 dev_type=None, cache_dir=None):
        # create context
        super(Context, self).__init__(devices, properties,
                                      dev_type, cache_dir)
        # initialize variable
        self.device_list = []
        self.program = None
        # initialize each device in this context
        for d in devices:
            self.device_list.append(d)
            d.context = self
            d.queues = []
            # create default queue
            d.create_queue()

    # endregion : Constructor

    # region : Properties

    @property
    def default_queue(self):
        return self.default_device.queues[0]

    @property
    def default_device(self):
        return self.device_list[0]

    @property
    def devices(self):
        return super(Context, self).devices

    @property
    def num_devices(self):
        return super(Context, self).num_devices

    @property
    def properties(self):
        return super(Context, self).properties

    @property
    def reference_count(self):
        return super(Context, self).reference_count

    # endregion

    # region : Public Methods

    def create_build_program(self, src=None, devices=None,
                             options=[], cache_dir=None,
                             src_file_name=None):
        # import
        from _program import Program
        import clip.cl
        # check src
        if src is None:
            src = open(src_file_name).read()
        # create program
        self.program = Program(self, src)
        # build program
        self.program.build(options, devices, cache_dir)

        clip.cl.program = self.program

        return self.program

    def compile_link_program(self, header_infos, source_infos,
                             build_options=[],
                             compile_options=[], link_options=[],
                             info_is_filename=True):
        # > import
        from _program import Program
        import clip.cl
        import os

        # > set environment variable to suppress caching
        os.environ['PYOPENCL_NO_CACHE'] = 'TRUE'

        # > create header programs
        from pyopencl._cffi import ffi
        headers = []
        for info in header_infos:
            if info_is_filename:
                routine = open(info[0]).read()
            else:
                routine = info[0]
            # >> create header program
            pgm = Program(self, routine)
            # >> get cstring
            head_name = ffi.new("char[]", info[1])
            # >> wrap header
            headers += ((head_name, pgm._prg),)

        # > create source programs and compile
        sources = []
        for info in source_infos:
            if info_is_filename:
                routine = open(info).read()
            else:
                routine = info
            # >> create program
            pgm = Program(self, routine)
            # >> try to compile
            pgm.compile(compile_options, headers=headers)
            # >> add program to sources
            sources += (pgm._prg,)
        # > try to link
        import pyopencl

        # self.program = pyopencl.link_program(self, sources, link_options)

        options_bytes = ''
        self.program = pyopencl.Program(
            pyopencl._Program.link(self, sources, options_bytes, None))

        clip.cl.program = self.program
        self.program.__class__ = Program
        return self.program

    def create_buffer(self, access_mode, size=0, hostbuf=None,
                      host_ptr_mode=mem_host_ptr_mode.DEFAULT):
        # import
        from _buffer import Buffer
        # create buffer
        buf = Buffer(self, access_mode | host_ptr_mode,
                     size, hostbuf)
        return buf

    # endregion : Public Methods

    pass  # patch for region
