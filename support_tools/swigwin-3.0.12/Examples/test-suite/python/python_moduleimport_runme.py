import python_moduleimport

if python_moduleimport.simple_function(99) != 99:
    raise RuntimeError("simple_function")

if python_moduleimport.extra_import_variable != "custom import of _python_moduleimport":
    raise RuntimeError("custom import")
