# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from urllib import request

import os
import tarfile
import shutil

url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

tmpGz = "tmp.tgz"
tmp_dir  = './tmp/'

tar_path_test       = './simple-examples/data/ptb.test.txt'
tar_path_train      = './simple-examples/data/ptb.train.txt'
tar_path_valid      = './simple-examples/data/ptb.valid.txt'


def append_eos(input_path, output_path):
    with open(input_path, 'r') as input_file, \
         open(output_path, 'w') as output_file:
        for line in input_file:
            line = line.strip()
            output_file.write(line + " <eos>\n")

if __name__=='__main__':

    if not os.path.isfile(tmpGz):
        request.urlretrieve(url, tmpGz)

    # extracting the files we need from the tarfile
    fileReader = tarfile.open(tmpGz, 'r') 
    fileReader.extract(tar_path_test,  path=tmp_dir)
    fileReader.extract(tar_path_train, path=tmp_dir)
    fileReader.extract(tar_path_valid, path=tmp_dir)

    append_eos(os.path.join(tmp_dir, tar_path_test),  'test.txt')
    append_eos(os.path.join(tmp_dir, tar_path_train), 'train.txt')
    append_eos(os.path.join(tmp_dir, tar_path_valid), 'valid.txt')

    fileReader.close()

    os.remove(tmpGz)
    shutil.rmtree(tmp_dir)
