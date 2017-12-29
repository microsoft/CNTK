from example import *

JvCreateJavaVM(None)
JvAttachCurrentThread(None, None)

e1 = Example(1)
e2 = Example(2)

print e1.Add(1, 2)
print e1.Add(1.0, 2.0)
e3 = e1.Add(e1, e2)
print e3.mPublicInt

print e1.Add("1", "2")

JvDetachCurrentThread()
