from rename_predicates import *

r = RenamePredicates(123)
r.MF_member_function()
r.MF_static_member_function()
r.MF_extend_function_before()
r.MF_extend_function_after()
GF_global_function()

if r.MV_member_variable != 123:
    raise RuntimeError("variable wrong")
r.MV_member_variable = 1234
if r.MV_member_variable != 1234:
    raise RuntimeError("variable wrong")

if cvar.RenamePredicates_MV_static_member_variable != 456:
    raise RuntimeError("variable wrong")
cvar.RenamePredicates_MV_static_member_variable = 4567
if cvar.RenamePredicates_MV_static_member_variable != 4567:
    raise RuntimeError("variable wrong")

if cvar.GV_global_variable != 789:
    raise RuntimeError("variable wrong")
cvar.GV_global_variable = 7890
if cvar.GV_global_variable != 7890:
    raise RuntimeError("variable wrong")

UC_UPPERCASE()
LC_lowercase()
TI_Title()
FU_FirstUpperCase()
FL_firstLowerCase()
CA_CamelCase()
LC_lowerCamelCase()
UC_under_case_it()

ex = ExtendCheck()
ex.MF_real_member1()
ex.MF_real_member2()
ex.EX_EXTENDMETHOD1()
ex.EX_EXTENDMETHOD2()
ex.EX_EXTENDMETHOD3()
