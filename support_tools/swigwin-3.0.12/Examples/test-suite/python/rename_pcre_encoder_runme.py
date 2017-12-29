from rename_pcre_encoder import *

s = SomeWidget()
s.put_borderWidth(3)
if s.get_borderWidth() != 3:
    raise RuntimeError("Border should be 3, not %d" % (s.get_borderWidth(),))

s.put_size(4, 5)
a = AnotherWidget()
a.DoSomething()

evt = wxEVTSomeEvent()
t = xUnchangedName()

if StartINSAneAndUNSAvoryTraNSAtlanticRaNSAck() != 42:
    raise RuntimeError("Unexpected result of renamed function call")
