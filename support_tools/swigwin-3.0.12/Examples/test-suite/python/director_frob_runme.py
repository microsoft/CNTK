from director_frob import *

foo = Bravo()
s = foo.abs_method()

if s != "Bravo::abs_method()":
    raise RuntimeError, s
