import sys
import numpy as np

def writeConvWeights(fname, cmapIn, cmapOut):
    w = np.eye(cmapOut, cmapIn)
    np.savetxt(fname, w, fmt = '%d', delimiter = ' ')

if __name__ == "__main__":
    cmapIn = int(sys.argv[1])
    cmapOut = int(sys.argv[2])
    fname = sys.argv[3]
    writeConvWeights(fname, cmapIn, cmapOut)