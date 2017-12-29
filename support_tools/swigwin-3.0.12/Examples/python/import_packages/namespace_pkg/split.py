import sys
# Package brave split into two paths.
# path2/brave/robin.py and path3/brave/_robin.so
sys.path.insert(0, 'path2')
sys.path.insert(0, 'path3')

from brave import robin

assert(robin.run() == "AWAY!")
