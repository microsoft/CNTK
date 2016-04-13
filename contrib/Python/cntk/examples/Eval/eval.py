import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np

from cntk import *

if (__name__ == "__main__"):

    X = constant(np.asarray([[2, 3], [4, 5]]))
    out = 2.5 * X + 100

    with Context('demo', clean_up=False) as ctx:
        result = ctx.eval(out)
        print(result)
        # outputs:
        # [[ 105.   107.5]
        #  [ 110.   112.5]]
