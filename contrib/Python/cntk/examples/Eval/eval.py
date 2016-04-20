import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

#TODO: re-write the example using the new facade

import numpy as np

from cntk import *

#TODO necessary fix of CNTK exe. When we do not use inputs (just constants), the 
# output of the write action has a missing line.
if (__name__ == "__main__"):

    X = constant(np.asarray([[2, 3], [4, 5]]))
    out = 2.5 * X + 100

    with Context('demo', clean_up=True) as ctx:
        result = ctx.eval(out)
        print(result)
        # outputs:
        # [[ 105.   107.5]
        #  [ 110.   112.5]]
