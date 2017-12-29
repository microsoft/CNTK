
import _enums

_enums.bar2(1)
_enums.bar3(1)
_enums.bar1(1)

if _enums.cvar.enumInstance != 2:
    raise RuntimeError

if _enums.cvar.Slap != 10:
    raise RuntimeError

if _enums.cvar.Mine != 11:
    raise RuntimeError

if _enums.cvar.Thigh != 12:
    raise RuntimeError
