import multi_import_a
import multi_import_b

x = multi_import_b.XXX()
if x.testx() != 0:
    raise RuntimeError

y = multi_import_b.YYY()
if y.testx() != 0:
    raise RuntimeError
if y.testy() != 1:
    raise RuntimeError

z = multi_import_a.ZZZ()
if z.testx() != 0:
    raise RuntimeError
if z.testz() != 2:
    raise RuntimeError
