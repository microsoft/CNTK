# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from six.moves.urllib import request
# from urllib import request

import os
import tarfile
import shutil

url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

tmptgz = "tmp.tgz"
tmpdir = './tmp/'

tar_path_test  = './simple-examples/data/ptb.test.txt'
tar_path_train = './simple-examples/data/ptb.train.txt'
tar_path_valid = './simple-examples/data/ptb.valid.txt'


def append_eos(input_path, output_path):
    with open(input_path, 'r') as input_file, \
         open(output_path, 'w') as output_file:
        for line in input_file:
            line = line.strip()
            output_file.write(line + " <eos>\n")

if __name__=='__main__':

    if not os.path.isfile(tmptgz):
        request.urlretrieve(url, tmptgz)

    # extracting the files we need from the tarfile
    fileReader = tarfile.open(tmptgz, 'r') 
    fileReader.extract(tar_path_test,  path=tmpdir)
    fileReader.extract(tar_path_train, path=tmpdir)
    fileReader.extract(tar_path_valid, path=tmpdir)

    append_eos(os.path.join(tmpdir, tar_path_test),  'test.txt')
    append_eos(os.path.join(tmpdir, tar_path_train), 'train.txt')
    append_eos(os.path.join(tmpdir, tar_path_valid), 'valid.txt')

    fileReader.close()

    os.remove(tmptgz)
    shutil.rmtree(tmpdir)
