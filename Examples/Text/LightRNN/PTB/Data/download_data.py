# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

import os
import tarfile
import shutil

download_url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
save_file = 'temp.tgz'

def get_filename(name):
    return './simple-examples/data/ptb.{}'.format(name)

def add_eos(input_path, output_path):
    with open(input_path, 'r') as input_file, \
         open(output_path, 'w') as output_file:
        for line in input_file:
            line = line.strip()
            output_file.write(line + " </s>\n")

if __name__=='__main__':
    if not os.path.isfile(save_file):
        urlretrieve(download_url, save_file)

    fileReader = tarfile.open(save_file, 'r') 
    for name in ['train.txt', 'test.txt', 'valid.txt']:
        filename = get_filename(name)
        fileReader.extract(filename, path='.')
        add_eos(filename, name)

    fileReader.close()
    os.remove(save_file)
    shutil.rmtree('./simple-examples')
