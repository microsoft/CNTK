import overload_bool

# Overloading bool, int, string
if overload_bool.overloaded(True) != "bool":
    raise RuntimeError("wrong!")
if overload_bool.overloaded(False) != "bool":
    raise RuntimeError("wrong!")

if overload_bool.overloaded(0) != "int":
    raise RuntimeError("wrong!")
if overload_bool.overloaded(1) != "int":
    raise RuntimeError("wrong!")
if overload_bool.overloaded(2) != "int":
    raise RuntimeError("wrong!")

if overload_bool.overloaded("1234") != "string":
    raise RuntimeError("wrong!")

# Test bool masquerading as int
if overload_bool.intfunction(True) != "int":
    raise RuntimeError("wrong!")
if overload_bool.intfunction(False) != "int":
    raise RuntimeError("wrong!")

# Test int masquerading as bool
# Not possible


#############################################

# Overloading bool, int, string
if overload_bool.overloaded_ref(True) != "bool":
    raise RuntimeError("wrong!")
if overload_bool.overloaded_ref(False) != "bool":
    raise RuntimeError("wrong!")

if overload_bool.overloaded_ref(0) != "int":
    raise RuntimeError("wrong!")
if overload_bool.overloaded_ref(1) != "int":
    raise RuntimeError("wrong!")
if overload_bool.overloaded_ref(2) != "int":
    raise RuntimeError("wrong!")

if overload_bool.overloaded_ref("1234") != "string":
    raise RuntimeError("wrong!")

# Test bool masquerading as int
if overload_bool.intfunction_ref(True) != "int":
    raise RuntimeError("wrong!")
if overload_bool.intfunction_ref(False) != "int":
    raise RuntimeError("wrong!")

# Test int masquerading as bool
# Not possible
