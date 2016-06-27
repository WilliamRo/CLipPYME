########################################################################
#
#   Created: June 26, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from utility import data_type


def get_machine_parameter(i):
    if data_type == np.float32:
        if i is 1:
            return data_type(1.19209290e-07)
        elif i is 2:
            return data_type(1.17549435e-38)
        else:
            return data_type(3.40282347e+38)
    elif data_type == np.float64:
        if i is 1:
            return data_type(2.2204460492503131e-16)
        elif i is 2:
            return data_type(2.2250738585072014e-308)
        else:
            return data_type(1.7976931348623157e+308)
