from cpp11_raw_string_literals import *

if cvar.L != 100:
    raise RuntimeError

if cvar.u8 != 100:
    raise RuntimeError

if cvar.u != 100:
    raise RuntimeError

if UStruct.U != 100:
    raise RuntimeError


if cvar.R != 100:
    raise RuntimeError

if cvar.LR != 100:
    raise RuntimeError

if cvar.u8R != 100:
    raise RuntimeError

if cvar.uR != 100:
    raise RuntimeError

if URStruct.UR != 100:
    raise RuntimeError


if cvar.aa != "Wide string":
    raise RuntimeError

if cvar.bb != "UTF-8 string":
    raise RuntimeError, cvar.wide

if cvar.xx != ")I'm an \"ascii\" \\ string.":
    raise RuntimeError, cvar.xx

if cvar.ee != ")I'm an \"ascii\" \\ string.":
    raise RuntimeError, cvar.ee

if cvar.ff != "I'm a \"raw wide\" \\ string.":
    raise RuntimeError, cvar.ff

if cvar.gg != "I'm a \"raw UTF-8\" \\ string.":
    raise RuntimeError, cvar.gg
