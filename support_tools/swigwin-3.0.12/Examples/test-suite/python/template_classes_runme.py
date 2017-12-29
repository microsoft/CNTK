from template_classes import *

# This test is just testing incorrect number of arguments/parameters checking

point = PointInt()

rectangle = RectangleInt()
rectangle.setPoint(point)
rectangle.getPoint()
RectangleInt.static_noargs()
RectangleInt.static_onearg(1)

fail = True
try:
    rectangle.setPoint()
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")


fail = True
try:
    rectangle.getPoint(0)
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    RectangleInt.static_noargs(0)
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    RectangleInt.static_onearg()
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")
