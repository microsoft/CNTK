from greenlet import greenlet

c = 0
def f1():
  global c
  for i in range(20):
    c += 1
    print("f1", c)
    gr2.switch()

def f2():
  global c
  for i in range(20):
    c += 1
    print("f2", c)
    gr1.switch()

gr1 = greenlet(f1)
gr2 = greenlet(f2)
gr1.switch()
print("done")