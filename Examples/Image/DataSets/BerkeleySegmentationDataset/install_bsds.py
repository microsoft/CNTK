import os
import bsds_utils as ut

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    ut.download_data("./Images", "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/images/plain/normal/color/")
    ut.prep_64("./Images", 64, 64, "./train64_LR", "./train64_HR", "./tests")
    ut.prep_224("./Images", 224, 224, "./train112", "./train224")