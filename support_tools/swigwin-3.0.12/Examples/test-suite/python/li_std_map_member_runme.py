import li_std_map_member

a = li_std_map_member.mapita()
a[1] = li_std_map_member.TestA()

if (a[1].i != 1):
    raise RuntimeError("a[1] != 1")

a[1].i = 2
if (a[1].i != 2):
    raise RuntimeError("a[1] != 2")
