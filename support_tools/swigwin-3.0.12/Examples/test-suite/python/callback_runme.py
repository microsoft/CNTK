import _callback
from callback import *

if foo(2) != 2:
    raise RuntimeError

if A_bar(2) != 4:
    raise RuntimeError

if foobar(3, _callback.foo) != foo(3):
    raise RuntimeError

if foobar(3, foo) != foo(3):
    raise RuntimeError

if foobar(3, A_bar) != A_bar(3):
    raise RuntimeError

if foobar(3, foof) != foof(3):
    raise RuntimeError

if foobar_i(3, foo_i) != foo_i(3):
    raise RuntimeError


if foobar_d(3.5, foo_d) != foo_d(3.5):
    raise RuntimeError

a = A()
if foobarm(3, a, A.foom_cb_ptr) != a.foom(3):
    raise RuntimeError
