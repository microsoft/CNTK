# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import sys
try:
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

# Add models here like this: (category, model_name, model_url)
models = (('Image Classification', 'AlexNet_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/AlexNet_ImageNet_CNTK.model'),
          ('Image Classification', 'AlexNet_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/AlexNet_ImageNet_Caffe.model'),
          ('Image Classification', 'InceptionV3_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model'),
          ('Image Classification', 'BNInception_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/BNInception_ImageNet_Caffe.model'),
          ('Image Classification', 'ResNet18_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model'),
		  ('Image Classification', 'ResNet_18', 'https://cntkbuildstorage.blob.core.windows.net/cntk-pretrained-model/ResNet_18.model'),
          ('Image Classification', 'ResNet34_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet34_ImageNet_CNTK.model'),
          ('Image Classification', 'ResNet50_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet50_ImageNet_CNTK.model'),
          ('Image Classification', 'ResNet101_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet101_ImageNet_CNTK.model'),
          ('Image Classification', 'ResNet152_ImageNet_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet152_ImageNet_CNTK.model'),
          ('Image Classification', 'ResNet20_CIFAR10_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet20_CIFAR10_CNTK.model'),
          ('Image Classification', 'ResNet110_CIFAR10_CNTK', 'https://www.cntk.ai/Models/CNTK_Pretrained/ResNet110_CIFAR10_CNTK.model'),
          ('Image Classification', 'ResNet50_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/ResNet50_ImageNet_Caffe.model'),
          ('Image Classification', 'ResNet101_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/ResNet101_ImageNet_Caffe.model'),
          ('Image Classification', 'ResNet152_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/ResNet152_ImageNet_Caffe.model'),
          ('Image Classification', 'VGG16_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model'),
          ('Image Classification', 'VGG19_ImageNet_Caffe', 'https://www.cntk.ai/Models/Caffe_Converted/VGG19_ImageNet_Caffe.model'),
          ('Image Object Detection', 'Fast-RCNN_grocery100', 'https://www.cntk.ai/Models/FRCN_Grocery/Fast-RCNN_grocery100.model'),
          ('Image Object Detection', 'Fast-RCNN_Pascal', 'https://www.cntk.ai/Models/FRCN_Pascal/Fast-RCNN.model'))

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
    if model_name.endswith('.model'):
        model_name = model_name[:-6]

    model = next((x for x in models if x[1]==model_name), None)
    if model is None:
        print("ERROR: Unknown model name '%s'." % model_name)
        list_available_models()
    else:
        download_model(model_name + '.model', model[2])

def list_available_models():
    print("\nAvailable models (for more information see Readme.md):")
    max_cat = max(len(x[1]) for x in models)
    max_name = max(len(x[1]) for x in models)
    print("{:<{width}}   {}".format('Model name', 'Category', width=max_name))
    print("{:-<{width}}   {:-<{width_cat}}".format('', '', width=max_name, width_cat=max_cat))
    for model in sorted(models):
        print("{:<{width}}   {}".format(model[1], model[0], width=max_name))

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print("Please provide a model name as the single argument. Usage:")
        print("    python download_model.py <model_name>")
        list_available_models()
    else:
        model_name = args[1]
        if model_name == 'list':
            list_available_models()
        else:
            download_model_by_name(model_name)
