"""
The script shows the usage of converter
"""
import sys
from cntk.contrib.model2cntk import ModelConverter

NETWORK_DICT = {
    'AlexNet': 'Classification/AlexNet_ImageNet/global.json',
    'NIN': 'Classification/NIN_ImageNet/global.json',
    'VGG16': 'Classification/VGG_ImageNet/global_vgg16.json',
    'VGG19': 'Classification/VGG_ImageNet/global_vgg19.json',
    'GoogLeNet': 'Classification/GoogLeNet_ImageNet/global.json',
    'ResNet50': 'Classification/ResNet_ImageNet/global_resnet50.json',
    'ResNet101': 'Classification/ResNet_ImageNet/global_resnet101.json',
    'ResNet152': 'Classification/ResNet_ImageNet/global_resnet152.json',
}

def convert_model(model_select):
    """
    Convert the model in Examples

    Args:
        model_select (string): the model in NETWORK_DICT
        out_path (string): where to save the converted model
    
    Return:
        None
    """
    ModelConverter.convert_model(model_select)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Please specify the converted Model')
        sys.exit()
    sys.stdout = open('log.txt', 'w')
    if sys.argv[1] == 'ALL':
        for key in NETWORK_DICT.keys():
            convert_model(NETWORK_DICT[key])
            
        sys.exit()
    if sys.argv[1] not in NETWORK_DICT:
        sys.stderr.write('Please select the networks in the example lists \nResNet50')
        sys.exit()
    else:
        convert_model(NETWORK_DICT[sys.argv[1]])
