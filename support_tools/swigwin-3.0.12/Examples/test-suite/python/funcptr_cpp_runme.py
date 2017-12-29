from funcptr_cpp import *

if call1(ADD_BY_VALUE, 10, 11) != 21:
    raise RuntimeError
if call2(ADD_BY_POINTER, 12, 13) != 25:
    raise RuntimeError
if call3(ADD_BY_REFERENCE, 14, 15) != 29:
    raise RuntimeError
if call1(ADD_BY_VALUE_C, 2, 3) != 5:
    raise RuntimeError
