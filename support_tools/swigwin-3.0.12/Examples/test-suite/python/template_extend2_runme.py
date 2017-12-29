import template_extend2

a = template_extend2.lBaz()
b = template_extend2.dBaz()

if a.foo() != "lBaz::foo":
    raise RuntimeError

if b.foo() != "dBaz::foo":
    raise RuntimeError
