from python_docstring import *
import inspect

def check(got, expected):
    expected_list = expected.split("\n")
    got_list = got.split("\n")

    if expected_list != got_list:
        raise RuntimeError("\n" + "Expected: " + str(expected_list) + "\n" + "Got     : " + str(got_list))

# When getting docstrings, use inspect.getdoc(x) instead of x.__doc__ otherwise the different options
# such as -O, -builtin, -classic produce different initial indentation.

check(inspect.getdoc(DocStrings.docstring1),
    "  line 1\n"
    "line 2\n"
    "\n"
    "\n"
    "\n"
    "line 3"
    )

check(inspect.getdoc(DocStrings.docstring2),
    "line 1\n"
    "  line 2\n"
    "\n"
    "\n"
    "\n"
    "  line 3"
    )

check(inspect.getdoc(DocStrings.docstring3),
    "line 1\n"
    "  line 2\n"
    "\n"
    "\n"
    "\n"
    "  line 3"
    )

check(inspect.getdoc(DocStrings.docstring4),
    "line 1\n"
    "  line 2\n"
    "\n"
    "\n"
    "\n"
    "  line 3"
    )

check(inspect.getdoc(DocStrings.docstring5),
    "line 1\n"
    "  line 2\n"
    "\n"
    "\n"
    "\n"
    "  line 3"
    )

check(inspect.getdoc(DocStrings.docstring6),
    "line 1\n"
    "  line 2\n"
    "\n"
    "\n"
    "\n"
    "  line 3"
    )

check(inspect.getdoc(DocStrings.docstring7),
    "line 1\n"
    "line 2\n"
    "line 3"
    )

check(inspect.getdoc(DocStrings.docstringA),
    "first line\n"
    "second line"
    )

check(inspect.getdoc(DocStrings.docstringB),
    "first line\n"
    "second line"
    )

check(inspect.getdoc(DocStrings.docstringC),
    "first line\n"
    "second line"
    )

# One line doc special case, use __doc__
check(DocStrings.docstringX.__doc__,
    "  one line docs"
    )

check(inspect.getdoc(DocStrings.docstringX),
    "one line docs"
    )

check(inspect.getdoc(DocStrings.docstringY),
    "one line docs"
    )
