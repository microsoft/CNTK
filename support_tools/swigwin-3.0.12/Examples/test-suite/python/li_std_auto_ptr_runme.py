from li_std_auto_ptr import *

k1 = makeKlassAutoPtr("first")
k2 = makeKlassAutoPtr("second")
if Klass_getTotal_count() != 2:
    raise "number of objects should be 2"

del k1
if Klass_getTotal_count() != 1:
    raise "number of objects should be 1"

if k2.getLabel() != "second":
    raise "wrong object label"

del k2
if Klass_getTotal_count() != 0:
    raise "no objects should be left"
