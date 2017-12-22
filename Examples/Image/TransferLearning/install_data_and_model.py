# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, sys


base_folder = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(base_folder, "..", "DataSets", "Flowers"))
from install_flowers import download_flowers_data
download_flowers_data()

sys.path.append(os.path.join(base_folder, "..", "DataSets", "Animals"))
from install_animals import download_animals_data
download_animals_data()

sys.path.append(os.path.join(base_folder, "..", "DataSets", "Grocery"))
from install_grocery import download_grocery_data
download_grocery_data()

sys.path.append(os.path.join(base_folder, "..", "..", "..", "PretrainedModels"))
from download_model import download_model_by_name
download_model_by_name("ResNet18_ImageNet_CNTK")

