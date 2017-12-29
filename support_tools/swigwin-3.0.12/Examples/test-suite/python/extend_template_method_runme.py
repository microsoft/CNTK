from extend_template_method import *


em = ExtendMe()

ret_double = em.do_stuff_double(1, 1.1)
if ret_double != 1.1:
    raise RuntimeError("double failed " + ret_double)
ret_string = em.do_stuff_string(1, "hello there")
if ret_string != "hello there":
    raise RuntimeError("string failed " + ret_string)

ret_double = em.do_overloaded_stuff(1.1)
if ret_double != 1.1:
    raise RuntimeError("double failed " + ret_double)
ret_string = em.do_overloaded_stuff("hello there")
if ret_string != "hello there":
    raise RuntimeError("string failed " + ret_string)

if ExtendMe.static_method(123) != 123:
  raise RuntimeError("static_method failed")

em2 = ExtendMe(123)

em = TemplateExtend()

ret_double = em.do_template_stuff_double(1, 1.1)
if ret_double != 1.1:
    raise RuntimeError("double failed " + ret_double)
ret_string = em.do_template_stuff_string(1, "hello there")
if ret_string != "hello there":
    raise RuntimeError("string failed " + ret_string)


ret_double = em.do_template_overloaded_stuff(1.1)
if ret_double != 1.1:
    raise RuntimeError("double failed " + ret_double)
ret_string = em.do_template_overloaded_stuff("hello there")
if ret_string != "hello there":
    raise RuntimeError("string failed " + ret_string)

if TemplateExtend.static_template_method(123) != 123:
  raise RuntimeError("static_template_method failed")

em2 = TemplateExtend(123)
