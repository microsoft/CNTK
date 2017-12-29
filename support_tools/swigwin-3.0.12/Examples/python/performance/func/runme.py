import sys
sys.path.append('..')
import harness


def proc(mod):
    x = mod.MyClass()
    for i in range(10000000):
        x.func()

harness.run(proc)
