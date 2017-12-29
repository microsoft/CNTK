import ret_by_value

a = ret_by_value.get_test()
if a.myInt != 100:
    raise RuntimeError

if a.myShort != 200:
    raise RuntimeError
