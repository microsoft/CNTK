#!/usr/bin/evn python
from cpp_static import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

if is_new_style_class(StaticFunctionTest):
    StaticFunctionTest.static_func()
    StaticFunctionTest.static_func_2(1)
    StaticFunctionTest.static_func_3(1, 2)
else:
    StaticFunctionTest().static_func()
    StaticFunctionTest().static_func_2(1)
    StaticFunctionTest().static_func_3(1, 2)

if is_python_builtin():
  if not StaticMemberTest.static_int == 99: raise RuntimeError("static_int not 99")
  StaticMemberTest.static_int = 10
  if not StaticMemberTest.static_int == 10: raise RuntimeError("static_int not 10")

  if not StaticBase.statty == 11: raise RuntimeError("statty not 11")
  if not StaticDerived.statty == 111: raise RuntimeError("statty not 111")
  StaticBase.statty = 22
  StaticDerived.statty = 222
  if not StaticBase.statty == 22: raise RuntimeError("statty not 22")
  if not StaticDerived.statty == 222: raise RuntimeError("statty not 222")

  # Restore
  StaticMemberTest.static_int = 99
  StaticBase.statty = 11
  StaticDerived.statty = 111

if not cvar.StaticMemberTest_static_int == 99: raise RuntimeError("cvar static_int not 99")
cvar.StaticMemberTest_static_int = 10
if not cvar.StaticMemberTest_static_int == 10: raise RuntimeError("cvar static_int not 10")

if not cvar.StaticBase_statty == 11: raise RuntimeError("cvar statty not 11")
if not cvar.StaticDerived_statty == 111: raise RuntimeError("cvar statty not 111")
cvar.StaticBase_statty = 22
cvar.StaticDerived_statty = 222
if not cvar.StaticBase_statty == 22: raise RuntimeError("cvar statty not 22")
if not cvar.StaticDerived_statty == 222: raise RuntimeError("cvar statty not 222")
