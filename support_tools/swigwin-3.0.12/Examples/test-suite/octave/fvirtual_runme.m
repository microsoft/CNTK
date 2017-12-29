fvirtual

sw = NodeSwitch();
n = Node();
i = sw.addChild(n);

if (i != 2)
  error("addChild")
endif

