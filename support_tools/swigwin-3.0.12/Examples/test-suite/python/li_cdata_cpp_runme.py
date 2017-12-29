
from li_cdata_cpp import *

s = "ABC abc"
m = malloc(256)
memmove(m, s)
ss = cdata(m, 7)
if ss != "ABC abc":
    raise "failed"
