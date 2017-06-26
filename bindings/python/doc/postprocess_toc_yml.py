# -*- coding: utf-8 -*-

# Forked from
#   https://github.com/MicrosoftDocs/cognitive-toolkit-docs-python/blob/master/ci_scripts/postprocess_toc_yml.py

from __future__ import print_function
import io
import shutil
import yaml

tocPath = r'.\_build\html\docfx_yaml\toc.yml'

def rewrite_yml(data):
    with io.open(tocPath, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

with open(tocPath, 'r') as stream:
    data_loaded = yaml.load(stream)

    # should only have one root node: cntk
    assert len(data_loaded) == 1

    cntk_node = data_loaded[0]
    if 'name' in cntk_node:
        print(cntk_node['name'])
        cntk_node['name'] = 'Reference'

    # change leave 2 node's name: remove 'cntk' prefix
    if 'items' in cntk_node:
        for item in cntk_node['items']:
            if 'name' in item:
                if item['name'].startswith('cntk.'):
                    item['name'] = item['name'][5:]
                    print('update old name to %s' % item['name'])

    rewrite_yml(data_loaded)
