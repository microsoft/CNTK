import sys
import time
import imp
from subprocess import *


def run(proc):

    try:
        mod = imp.find_module(sys.argv[1])
        mod = imp.load_module(sys.argv[1], *mod)

        t1 = time.clock()
        proc(mod)
        t2 = time.clock()
        print "%s took %f seconds" % (mod.__name__, t2 - t1)

    except IndexError:
        proc = Popen(
            [sys.executable, 'runme.py', 'Simple_baseline'], stdout=PIPE)
        (stdout, stderr) = proc.communicate()
        print stdout

        proc = Popen(
            [sys.executable, 'runme.py', 'Simple_optimized'], stdout=PIPE)
        (stdout, stderr) = proc.communicate()
        print stdout

        proc = Popen(
            [sys.executable, 'runme.py', 'Simple_builtin'], stdout=PIPE)
        (stdout, stderr) = proc.communicate()
        print stdout
