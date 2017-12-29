import python_pickle

import pickle
import sys

def is_new_style_class(cls):
    return hasattr(cls, "__class__")

def check(p):
    msg = p.msg
    if msg != "hi there":
        raise RuntimeError("Bad, got: " + msg)

if not is_new_style_class(python_pickle.PickleMe):
    sys.exit(0)

python_pickle.cvar.debug = False

p = python_pickle.PickleMe("hi there")
check(p)

r = p.__reduce__()
if python_pickle.cvar.debug:
    print "__reduce__ returned:", r
pickle_string = pickle.dumps(p)
newp = pickle.loads(pickle_string)
check(newp)

# Not yet working... some crash and others are not producing a sensible "can't be pickled" error
#nfp = python_pickle.NotForPickling("no no")
#print nfp.__reduce__()
#pickle_string = pickle.dumps(nfp)
#print pickle_string
#newp = pickle.loads(pickle_string)
#print newp.msg
