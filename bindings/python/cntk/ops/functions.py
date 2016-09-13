from cntk.cntk_py import Function
from cntk import DATATYPE
from cntk.graph import TensorOpsMixin

class Function(Function, TensorOpsMixin):
    pass
