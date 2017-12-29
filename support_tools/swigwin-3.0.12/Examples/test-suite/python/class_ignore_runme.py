import class_ignore

a = class_ignore.Bar()

if class_ignore.do_blah(a) != "Bar::blah":
    raise RuntimeError
