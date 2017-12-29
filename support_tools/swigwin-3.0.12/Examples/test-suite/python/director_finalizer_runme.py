from director_finalizer import *


class MyFoo(Foo):

    def __del__(self):
        self.orStatus(2)
        try:
            Foo.__del__(self)
        except:
            pass


resetStatus()

a = MyFoo()
del a

if getStatus() != 3:
    raise RuntimeError

resetStatus()

a = MyFoo()
launder(a)

if getStatus() != 0:
    raise RuntimeError

del a

if getStatus() != 3:
    raise RuntimeError

resetStatus()

a = MyFoo().__disown__()
deleteFoo(a)

if getStatus() != 3:
    raise RuntimeError

resetStatus()

a = MyFoo().__disown__()
deleteFoo(launder(a))

if getStatus() != 3:
    raise RuntimeError

resetStatus()
