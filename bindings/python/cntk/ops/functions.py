from cntk import Function
from cntk import DATATYPE
from cntk.graph import TensorOpsMixin

class Function(Function, TensorOpsMixin):
    pass
