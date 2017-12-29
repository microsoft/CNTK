import template_extend1

a = template_extend1.lBaz()
b = template_extend1.dBaz()

if a.foo() != "lBaz::foo":
    raise RuntimeError

if b.foo() != "dBaz::foo":
    raise RuntimeError
