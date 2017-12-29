from nested_template_base import *


ois = InnerS(123)
oic = InnerC()

# Check base method is available
if (oic.outer(ois).val != 123):
    raise RuntimeError("Wrong value calling outer")

# Check non-derived class using base class
if (oic.innerc().outer(ois).val != 123):
    raise RuntimeError("Wrong value calling innerc")
