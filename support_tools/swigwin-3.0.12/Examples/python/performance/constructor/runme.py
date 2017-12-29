import sys
sys.path.append('..')
import harness


def proc(mod):
    for i in range(1000000):
        x = mod.MyClass()

harness.run(proc)
