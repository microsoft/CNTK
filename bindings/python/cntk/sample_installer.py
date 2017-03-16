# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import sys
import os
import warnings
import textwrap
import zipfile
import string
import pip
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

from cntk import __version__

def module_is_unreleased():
    return __version__.endswith('+')

def default_sample_dir():
    version = __version__
    if module_is_unreleased():
        version = version[:-1]
    version = version.replace('.', '-')
    return 'CNTK-Samples-%s' % (version)

def default_sample_url():
    return 'https://cntk.ai/Samples/%s.zip' % (default_sample_dir())

def install_samples(url=None, directory=None, quiet=False):
    '''
    Fetch the CNTK samples from a URL, extract to local directory, and install
    Python package requirements.

    Args:
        url: the URL of the sample zip file (if passed ``None``, which is default,
         the sample URL matching the CNTK module release is chosen)
        directory: the directory to extract the samples into (if passed
         ``None``, which is default, a name matching the CNTK module release is
         chosen, ``CNTK-Samples-<version>``)
        quiet: whether to suppress all output (defaults to ``False``)
    '''

    def show_message(text):
        if not quiet:
            print(textwrap.fill(text, subsequent_indent=' ' * 4), "\n")

    if not url:
        if module_is_unreleased():
            show_message('WARNING: You are not on a released CNTK module version. '
                'When running into problems running samples, please check '
                'https://github.com/Microsoft/CNTK for updated versions')
        url = default_sample_url()

    if not directory:
        directory = default_sample_dir()

    if os.path.lexists(directory) and not (os.path.isdir(directory) and len(os.listdir(directory)) == 0):
        raise ValueError('directory %s must be non-existing or point to an empty directory' % (directory))

    zip_filename = None
    try:
        show_message('INFO: retrieving ' + url)
        zip_filename, message = urlretrieve(url)
        show_message('INFO: unzipping to directory ' + directory)
        with zipfile.ZipFile(zip_filename, 'r') as zip_file:
            zip_file.extractall(directory)
        requirements_file = os.path.join(directory, 'requirements.txt')
        if os.path.isfile(requirements_file):
            show_message('INFO: installing requirements')
            pip.main(['install', '-r', requirements_file])
        else:
            show_message('WARNING: file %s does not exist, modules to run the samples may be missing'
                % (requirements_file))
    finally:
        if zip_filename: os.remove(zip_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CNTK Sample Installer")
    parser.add_argument('-u', '--url',
        help='the url of the zip file to download (default: %(default)s)',
        default=default_sample_url())
    parser.add_argument('-d', '--directory',
            help='the directory to extract the samples into (default: %(default)s)',
            default=default_sample_dir())
    parser.add_argument('-q', '--quiet', action='store_true',
            help='suppress output (default: %(default)s)', default=False)
    
    options = parser.parse_args(sys.argv[1:])

    install_samples(options.url, options.directory, options.quiet)
