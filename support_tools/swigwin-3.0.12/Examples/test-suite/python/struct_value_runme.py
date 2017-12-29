import struct_value

b = struct_value.Bar()

b.a.x = 3
if b.a.x != 3:
    raise RuntimeError

b.b.x = 3
if b.b.x != 3:
    raise RuntimeError


# Test dynamically added attributes - Github pull request #320
b.added = 123

if b.added != 123:
    raise RuntimeError("Wrong attribute value")

if not b.__dict__.has_key("added"):
    raise RuntimeError("Missing added attribute in __dict__")


class PyBar(struct_value.Bar):

    def __init__(self):
        self.extra = "hi"
        struct_value.Bar.__init__(self)

pybar = PyBar()
if not pybar.__dict__.has_key("extra"):
    raise RuntimeError("Missing extra attribute in __dict__")
if pybar.extra != "hi":
    raise RuntimeError("Incorrect attribute value for extra")
