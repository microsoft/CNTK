# Note that this test is also used by python_default_args_runme.py hence
# the use of __main__ and the run function


def is_new_style_class(cls):
    return hasattr(cls, "__class__")


def run(module_name):
    default_args = __import__(module_name)
    ec = default_args.EnumClass()
    if not ec.blah():
        raise RuntimeError("EnumClass::blah() default arguments don't work")

    de = default_args.DerivedEnumClass()
    de.accelerate()
    de.accelerate(default_args.EnumClass.SLOW)

    if default_args.Statics_staticMethod() != 60:
        raise RuntimeError

    if default_args.cfunc1(1) != 2:
        raise RuntimeError

    if default_args.cfunc2(1) != 3:
        raise RuntimeError

    if default_args.cfunc3(1) != 4:
        raise RuntimeError

    f = default_args.Foo()

    f.newname()
    f.newname(1)

    if f.double_if_void_ptr_is_null(2, None) != 4:
        raise RuntimeError

    if f.double_if_void_ptr_is_null(3) != 6:
        raise RuntimeError

    if f.double_if_handle_is_null(4, None) != 8:
        raise RuntimeError

    if f.double_if_handle_is_null(5) != 10:
        raise RuntimeError

    if f.double_if_dbl_ptr_is_null(6, None) != 12:
        raise RuntimeError

    if f.double_if_dbl_ptr_is_null(7) != 14:
        raise RuntimeError

    try:
        f = default_args.Foo(1)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::Foo ignore is not working")

    try:
        f = default_args.Foo(1, 2)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::Foo ignore is not working")

    try:
        f = default_args.Foo(1, 2, 3)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::Foo ignore is not working")

    try:
        m = f.meth(1)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::meth ignore is not working")

    try:
        m = f.meth(1, 2)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::meth ignore is not working")

    try:
        m = f.meth(1, 2, 3)
        error = 1
    except:
        error = 0
    if error:
        raise RuntimeError("Foo::meth ignore is not working")

    if is_new_style_class(default_args.Klass):
        Klass_inc = default_args.Klass.inc
    else:
        Klass_inc = default_args.Klass_inc

    if Klass_inc(100, default_args.Klass(22)).val != 122:
        raise RuntimeError("Klass::inc failed")

    if Klass_inc(100).val != 99:
        raise RuntimeError("Klass::inc failed")

    if Klass_inc().val != 0:
        raise RuntimeError("Klass::inc failed")

    default_args.trickyvalue1(10)
    default_args.trickyvalue1(10, 10)
    default_args.trickyvalue2(10)
    default_args.trickyvalue2(10, 10)
    default_args.trickyvalue3(10)
    default_args.trickyvalue3(10, 10)
    default_args.seek()
    default_args.seek(10)

    if default_args.slightly_off_square(10) != 102:
        raise RuntimeError

    if default_args.slightly_off_square() != 291:
        raise RuntimeError

    # It is difficult to test the python:cdefaultargs feature properly as -builtin
    # and -fastproxy do not use the Python layer for default args
    if default_args.CDA().cdefaultargs_test1() != 1:
        raise RuntimeError

    if default_args.CDA().cdefaultargs_test2() != 1:
        raise RuntimeError

    if default_args.chartest1() != 'x':
        raise RuntimeError

    if default_args.chartest2() != '\0':
        raise RuntimeError

    if default_args.chartest3() != '\1':
        raise RuntimeError

    if default_args.chartest4() != '\n':
        raise RuntimeError

    if default_args.chartest5() != 'B':
        raise RuntimeError

    if default_args.chartest6() != 'C':
        raise RuntimeError

if __name__ == "__main__":
    run('default_args')
