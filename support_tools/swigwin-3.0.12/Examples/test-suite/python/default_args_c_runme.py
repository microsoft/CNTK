import default_args_c

if default_args_c.foo1() != 1:
  raise RuntimeError("failed")
if default_args_c.foo43() != 43:
  raise RuntimeError("failed")
