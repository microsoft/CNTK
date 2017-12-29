from autodoc import *
import sys


def check(got, expected, expected_builtin=None, skip=False):
    if not skip:
        expect = expected
        if is_python_builtin() and expected_builtin != None:
            expect = expected_builtin
        if expect != got:
            raise RuntimeError(
                "\n" + "Expected: [" + str(expect) + "]\n" + "Got     : [" + str(got) + "]")


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

def is_fastproxy(module):
    return "new_instancemethod" in module

if not is_new_style_class(A):
    # Missing static methods make this hard to test... skip if -classic is
    # used!
    sys.exit(0)

if is_fastproxy(dir()):
    # Detect when -fastproxy is specified and skip test as it changes the function names making it
    # hard to test... skip until the number of options are reduced in SWIG-3.1 and autodoc is improved
    sys.exit(0)

# skip builtin check - the autodoc is missing, but it probably should not be
skip = True

check(A.__doc__, "Proxy of C++ A class.", "::A")
check(A.funk.__doc__, "just a string.")
check(A.func0.__doc__,
      "func0(self, arg2, hello) -> int",
      "func0(arg2, hello) -> int")
check(A.func1.__doc__,
      "func1(A self, short arg2, Tuple hello) -> int",
      "func1(short arg2, Tuple hello) -> int")
check(A.func2.__doc__,
      "\n"
      "        func2(self, arg2, hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func2(arg2, hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )
check(A.func3.__doc__,
      "\n"
      "        func3(A self, short arg2, Tuple hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func3(short arg2, Tuple hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )

check(A.func0default.__doc__,
      "\n"
      "        func0default(self, e, arg3, hello, f=2) -> int\n"
      "        func0default(self, e, arg3, hello) -> int\n"
      "        ",
      "\n"
      "func0default(e, arg3, hello, f=2) -> int\n"
      "func0default(e, arg3, hello) -> int\n"
      ""
      )
check(A.func1default.__doc__,
      "\n"
      "        func1default(A self, A e, short arg3, Tuple hello, double f=2) -> int\n"
      "        func1default(A self, A e, short arg3, Tuple hello) -> int\n"
      "        ",
      "\n"
      "func1default(A e, short arg3, Tuple hello, double f=2) -> int\n"
      "func1default(A e, short arg3, Tuple hello) -> int\n"
      ""
      )
check(A.func2default.__doc__,
      "\n"
      "        func2default(self, e, arg3, hello, f=2) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg3: short\n"
      "        hello: int tuple[2]\n"
      "        f: double\n"
      "\n"
      "        func2default(self, e, arg3, hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg3: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func2default(e, arg3, hello, f=2) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg3: short\n"
      "hello: int tuple[2]\n"
      "f: double\n"
      "\n"
      "func2default(e, arg3, hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg3: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )
check(A.func3default.__doc__,
      "\n"
      "        func3default(A self, A e, short arg3, Tuple hello, double f=2) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg3: short\n"
      "        hello: int tuple[2]\n"
      "        f: double\n"
      "\n"
      "        func3default(A self, A e, short arg3, Tuple hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg3: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func3default(A e, short arg3, Tuple hello, double f=2) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg3: short\n"
      "hello: int tuple[2]\n"
      "f: double\n"
      "\n"
      "func3default(A e, short arg3, Tuple hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg3: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )

check(A.func0static.__doc__,
      "\n"
      "        func0static(e, arg2, hello, f=2) -> int\n"
      "        func0static(e, arg2, hello) -> int\n"
      "        ",
      "\n"
      "func0static(e, arg2, hello, f=2) -> int\n"
      "func0static(e, arg2, hello) -> int\n"
      ""
      )
check(A.func1static.__doc__,
      "\n"
      "        func1static(A e, short arg2, Tuple hello, double f=2) -> int\n"
      "        func1static(A e, short arg2, Tuple hello) -> int\n"
      "        ",
      "\n"
      "func1static(A e, short arg2, Tuple hello, double f=2) -> int\n"
      "func1static(A e, short arg2, Tuple hello) -> int\n"
      ""
      )
check(A.func2static.__doc__,
      "\n"
      "        func2static(e, arg2, hello, f=2) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "        f: double\n"
      "\n"
      "        func2static(e, arg2, hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func2static(e, arg2, hello, f=2) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "f: double\n"
      "\n"
      "func2static(e, arg2, hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )
check(A.func3static.__doc__,
      "\n"
      "        func3static(A e, short arg2, Tuple hello, double f=2) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "        f: double\n"
      "\n"
      "        func3static(A e, short arg2, Tuple hello) -> int\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        e: A *\n"
      "        arg2: short\n"
      "        hello: int tuple[2]\n"
      "\n"
      "        ",
      "\n"
      "func3static(A e, short arg2, Tuple hello, double f=2) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "f: double\n"
      "\n"
      "func3static(A e, short arg2, Tuple hello) -> int\n"
      "\n"
      "Parameters\n"
      "----------\n"
      "e: A *\n"
      "arg2: short\n"
      "hello: int tuple[2]\n"
      "\n"
      ""
      )

if sys.version_info[0:2] > (2, 4):
    # Python 2.4 does not seem to work
    check(A.variable_a.__doc__,
          "A_variable_a_get(self) -> int",
          "A.variable_a"
          )
    check(A.variable_b.__doc__,
          "A_variable_b_get(A self) -> int",
          "A.variable_b"
          )
    check(A.variable_c.__doc__,
          "\n"
          "A_variable_c_get(self) -> int\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "self: A *\n"
          "\n",
          "A.variable_c"
          )
    check(A.variable_d.__doc__,
          "\n"
          "A_variable_d_get(A self) -> int\n"
          "\n"
          "Parameters\n"
          "----------\n"
          "self: A *\n"
          "\n",
          "A.variable_d"
          )

check(B.__doc__,
      "Proxy of C++ B class.",
      "::B"
      )
check(C.__init__.__doc__, "__init__(self, a, b, h) -> C", None, skip)
check(D.__init__.__doc__,
      "__init__(D self, int a, int b, Hola h) -> D", None, skip)
check(E.__init__.__doc__,
      "\n"
      "        __init__(self, a, b, h) -> E\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        a: special comment for parameter a\n"
      "        b: another special comment for parameter b\n"
      "        h: enum Hola\n"
      "\n"
      "        ", None, skip
      )
check(F.__init__.__doc__,
      "\n"
      "        __init__(F self, int a, int b, Hola h) -> F\n"
      "\n"
      "        Parameters\n"
      "        ----------\n"
      "        a: special comment for parameter a\n"
      "        b: another special comment for parameter b\n"
      "        h: enum Hola\n"
      "\n"
      "        ", None, skip
      )

check(B.funk.__doc__,
      "funk(B self, int c, int d) -> int",
      "funk(int c, int d) -> int")
check(funk.__doc__, "funk(A e, short arg2, int c, int d) -> int")
check(funkdefaults.__doc__,
      "\n"
      "    funkdefaults(A e, short arg2, int c, int d, double f=2) -> int\n"
      "    funkdefaults(A e, short arg2, int c, int d) -> int\n"
      "    ",
      "\n"
      "funkdefaults(A e, short arg2, int c, int d, double f=2) -> int\n"
      "funkdefaults(A e, short arg2, int c, int d) -> int\n"
      ""
      )

check(func_input.__doc__, "func_input(int * INPUT) -> int")
check(func_output.__doc__, "func_output() -> int")
check(func_inout.__doc__, "func_inout(int * INOUT) -> int")
check(func_cb.__doc__, "func_cb(int c, int d) -> int")
check(banana.__doc__, "banana(S a, S b, int c, Integer d)")
