from li_cmalloc import *

p = malloc_int()
free_int(p)

ok = 0
try:
    p = calloc_int(-1)
    free_int(p)
except:
    ok = 1

if ok != 1:
    raise RuntimeError
