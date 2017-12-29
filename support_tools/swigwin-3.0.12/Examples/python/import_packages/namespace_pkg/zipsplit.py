import sys
# Package brave split into two paths.
# brave/robin.py (in path4.zip) and path3/brave/_robin.so
sys.path.insert(0, 'path4.zip')
sys.path.insert(0, 'path3')

from brave import robin

assert(robin.run() == "AWAY!")
