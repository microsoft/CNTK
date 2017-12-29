from global_functions import *


def check(a, b):
    if a != b:
        raise RuntimeError("Failed: " + str(a) + " != " + str(b))
global_void()
check(global_one(1), 1)
check(global_two(2, 2), 4)

fail = True
try:
    global_void(1)
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    global_one()
except TypeError, e:
    fail = False
if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    global_one(2, 2)
except TypeError, e:
    fail = False

if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    global_two(1)
except TypeError, e:
    fail = False

if fail:
    raise RuntimeError("argument count check failed")

fail = True
try:
    global_two(3, 3, 3)
except TypeError, e:
    fail = False

if fail:
    raise RuntimeError("argument count check failed")
