import sys
import numpy as np

def writeConvWeights(fname, cmapIn):
    cmapOut = 2 * cmapIn
    w = np.eye(cmapOut, cmapIn)
    np.savetxt(fname, w, fmt = '%d', delimiter = ' ')

if __name__ == "__main__":
    cmapIn = int(sys.argv[1])
    fname = sys.argv[2]
    writeConvWeights(fname, cmapIn)
