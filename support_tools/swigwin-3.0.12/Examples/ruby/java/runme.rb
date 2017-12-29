require 'example'

Example.JvCreateJavaVM(nil)
Example.JvAttachCurrentThread(nil, nil)

e1 = Example::Example.new(1)
e2 = Example::Example.new(2)

print e1.Add(1,2),"\n"
print e1.Add(1.0,2.0),"\n"
e3 = e1.Add(e1,e2)
print e3.mPublicInt,"\n"


print e1.Add("1","2"),"\n"

Example.JvDetachCurrentThread()

