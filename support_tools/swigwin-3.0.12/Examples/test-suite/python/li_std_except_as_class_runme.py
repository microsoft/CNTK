from li_std_except_as_class import *

# This test is expected to fail with -builtin option.
# Throwing builtin classes as exceptions not supported
if is_python_builtin():
    try:
        test_domain_error()
    except RuntimeError:
        pass
    try:
        test_domain_error()
    except RuntimeError:
        pass
    try:
        test_domain_error()
    except RuntimeError:
        pass
else:
    # std::domain_error hierarchy
    try:
        test_domain_error()
    except domain_error:
        pass
    try:
        test_domain_error()
    except logic_error:
        pass
    try:
        test_domain_error()
    except exception:
        pass
