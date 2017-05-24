#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from __future__ import print_function
import os
import ucf11_utils as ut

if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    ut.download_and_extract('http://crcv.ucf.edu/data/YouTube_DataSet_Annotated.zip')
    print ('Writing train and test CSV file...')
    ut.generate_and_save_labels()
    print ('Done.')
