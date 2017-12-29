from fvirtual import *

sw = NodeSwitch()
n = Node()
i = sw.addChild(n)

if i != 2:
    raise RuntimeError, "addChild"
