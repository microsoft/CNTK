# Test case from bug #1506850
#"When threading is enabled, the interpreter will infinitely wait on a mutex the second
# time this type of extended method is called. Attached is an example
# program that waits on the mutex to be unlocked."

from director_extend import *


class MyObject(SpObject):

    def __init__(self):
        SpObject.__init__(self)
        return

    def getFoo(self):
        return 123

m = MyObject()
if m.dummy() != 666:
    raise RuntimeError, "1st call"
if m.dummy() != 666:                        # Locked system
    raise RuntimeError, "2nd call"
