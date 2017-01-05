# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import os.path
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
   
def download_model(model_file_name, model_url):
    model_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(model_dir, model_file_name)
    if not os.path.exists(filename):
        print('Downloading model from ' + model_url + ', may take a while...')
        urlretrieve(model_url, filename)
        print('Saved model as ' + filename)
    else:
        print('CNTK model already available at ' + filename)
    
def download_model_by_name(model_name):
    model_file_name = model_name + '.model'
    if model_name == 'AlexNet':
        download_model(model_file_name, 'https://www.cntk.ai/Models/AlexNet/AlexNet.model')
    elif model_name == 'AlexNetBS':
        download_model(model_file_name, 'https://www.cntk.ai/Models/AlexNet/AlexNetBS.model')
    elif model_name == 'ResNet_18':
        # TODO: store new ResNet models in cntk.ai/Models/ResNet for consistency
        download_model(model_file_name, 'https://www.cntk.ai/Models/ResNet/ResNet_18.model')
    else:
        print("WARNING: Unknown model name '%s'." % model_name)
