import _profiletest       
import profiletest       

a = profiletest.A()
print a
print a.this

b = profiletest.B()
fn = b.fn
i = 50000
while i:
  a = fn(a) #1
  a = fn(a) #2
  a = fn(a) #3
  a = fn(a) #4
  a = fn(a) #5
  a = fn(a) #6
  a = fn(a) #7
  a = fn(a) #8
  a = fn(a) #9
  a = fn(a) #10
  a = fn(a) #1
  a = fn(a) #2
  a = fn(a) #3
  a = fn(a) #4
  a = fn(a) #5
  a = fn(a) #6
  a = fn(a) #7
  a = fn(a) #8
  a = fn(a) #9
  a = fn(a) #20
  i -= 1
