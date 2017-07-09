import re
import json
import sys
import platform


def camel_to_snake(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    string = re.sub('(.)([0-9]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


# Only used in Python 2
def unicode_to_utf(contents):
    if isinstance(contents, dict):
        return {unicode_to_utf(key): unicode_to_utf(value) for key, value in contents.iteritems()}
    elif isinstance(contents, list):
        return [unicode_to_utf(element) for element in contents]
    elif isinstance(contents, unicode):
        return contents.encode('utf-8')
    else:
        return contents


def json_parser(path):
    with open(path, 'r') as conf:
        conf = json.JSONDecoder().raw_decode(conf.read())[0]
    # to support python2/3 in both unicode and utf-8
    python_version = [int(v) for v in platform.python_version().split('.')]
    if python_version[0] < 3:
        conf = unicode_to_utf(conf)
    return conf
