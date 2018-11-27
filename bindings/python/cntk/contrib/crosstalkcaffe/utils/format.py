# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
'''
The script contains some basic format ops
'''
import re
import json
import platform


def camel_to_snake(camel_string):
    '''
     Convert a string from camel style to snake style

    Args:
        camel_string (str): the camel style string

    Return:
        (str): the snake style string
    '''
    camel_string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_string)
    camel_string = re.sub('(.)([0-9]+)', r'\1_\2', camel_string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', camel_string).lower()


# Only used in Python 2
def _unicode_to_utf(contents):
    if isinstance(contents, dict):
        return {_unicode_to_utf(key): _unicode_to_utf(value) for key, value in contents.iteritems()}
    elif isinstance(contents, list):
        return [_unicode_to_utf(element) for element in contents]
    elif isinstance(contents, unicode):
        return contents.encode('utf-8')
    else:
        return contents


def json_parser(path):
    '''
     Parse a json file into dict

    Args:
        path: the path to json file

    Return:
        (dict): the parsed dict
    '''
    with open(path, 'r') as conf:
        conf = json.JSONDecoder().raw_decode(conf.read())[0]
    # to support python2/3 in both unicode and utf-8
    python_version = [int(v) for v in platform.python_version().split('.')]
    if python_version[0] < 3:
        conf = _unicode_to_utf(conf)
    return conf
