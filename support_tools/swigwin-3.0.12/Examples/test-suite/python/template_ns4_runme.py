from template_ns4 import *

d = make_Class_DD()
if d.test() != "test":
    raise RuntimeError
