import memberin_extend_c

t = memberin_extend_c.Person()
t.name = "Fred Bloggs"
if t.name != "FRED BLOGGS":
    raise RuntimeError("name wrong")
