import threads_exception

t = threads_exception.Test()
try:
    t.unknown()
except RuntimeError, e:
    pass

try:
    t.simple()
except RuntimeError, e:
    if e.args[0] != 37:
        raise RuntimeError

try:
    t.message()
except RuntimeError, e:
    if e.args[0] != "I died.":
        raise RuntimeError

# This is expected fail with -builtin option
# Throwing builtin classes as exceptions not supported
if not threads_exception.is_python_builtin():
    try:
        t.hosed()
    except threads_exception.Exc, e:
        code = e.code
        if code != 42:
            raise RuntimeError, "bad... code: %d" % code
        msg = e.msg
        if msg != "Hosed":
            raise RuntimeError, "bad... msg: '%s' len: %d" % (msg, len(msg))

for i in range(1, 4):
    try:
        t.multi(i)
    except RuntimeError, e:
        pass
    except threads_exception.Exc, e:
        pass
