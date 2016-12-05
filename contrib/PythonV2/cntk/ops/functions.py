from cntk import cntk_py
from cntk import DATATYPE
from cntk.graph import TensorOpsMixin

class Function(cntk_py.Function, TensorOpsMixin):
    pass
