from cpp11_thread_local import *

t = ThreadLocals()
if t.stval != 11:
    raise RuntimeError
if t.tsval != 22:
    raise RuntimeError
if t.tscval99 != 99:
    raise RuntimeError

cvar.etval = -11
if cvar.etval != -11:
    raise RuntimeError

cvar.stval = -22
if cvar.stval != -22:
    raise RuntimeError

cvar.tsval = -33
if cvar.tsval != -33:
    raise RuntimeError

cvar.etval = -44
if cvar.etval != -44:
    raise RuntimeError

cvar.teval = -55
if cvar.teval != -55:
    raise RuntimeError

cvar.ectval = -66
if cvar.ectval != -66:
    raise RuntimeError

cvar.ecpptval = -66
if cvar.ecpptval != -66:
    raise RuntimeError
