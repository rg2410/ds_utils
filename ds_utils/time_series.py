# -*- coding: utf-8 -*-
"""
Time Series tools

Author: Rafael Fernandes
"""

import numpy as np

def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])