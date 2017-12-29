from template_type_namespace import *

if type(foo()[0]) != type(""):
    raise RuntimeError
