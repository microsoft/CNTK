# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
from json import dumps

abs_path = os.path.dirname(os.path.realpath(__file__))
grocery_map_file = os.path.join(abs_path, "..", "..", "..", "DataSets", "Grocery", "class_map.txt")
pascal_map_file = os.path.join(abs_path, "..", "..", "..", "DataSets", "Pascal", "mappings", "class_map.txt")

def flat_tree_grocery_str(map_file_json_str):
    return '[' \
    '   {"id": 0, "children": [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "syn": "Synset(\'physical_entity.n.01\')", "strings": ["physical_entity.n.01"], "cls_maps": []}, ' \
    '   {"id": 1, "children": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 2, "children": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 3, "children": [], "syn": "Synset(\'juice.n.01\')", "strings": ["orangeJuice"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 4, "children": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 5, "children": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gerkin"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 6, "children": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggBox"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 7, "children": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 8, "children": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["joghurt"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 9, "children": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 10, "children": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 11, "children": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 12, "children": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 13, "children": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 14, "children": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 15, "children": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 16, "children": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": [' + map_file_json_str + ']}' \
    ']'

def tree_grocery_str(map_file_json_str):
    return '[' \
    '   {"id": 0, "children": [21, 17], "syn": "Synset(\'physical_entity.n.01\')", "strings": ["physical_entity.n.01"], "cls_maps": []}, ' \
    '   {"id": 1, "children": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 2, "children": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 3, "children": [], "syn": "Synset(\'juice.n.01\')", "strings": ["orangeJuice"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 4, "children": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 5, "children": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gerkin"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 6, "children": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggBox"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 7, "children": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 8, "children": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["joghurt"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 9, "children": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 10, "children": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 11, "children": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 12, "children": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 13, "children": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 14, "children": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 15, "children": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": [' + map_file_json_str + ']}, ' \
    '   {"id": 16, "children": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": [' + map_file_json_str + ']},' \
    '   {"id": 17, "children": [20, 19, 18], "syn": null, "strings": ["packaged"], "cls_maps": []}, ' \
    '   {"id": 18, "children": [11, 7, 4, 3, 2, 1], "syn": null, "strings": ["bottled"], "cls_maps": []}, ' \
    '   {"id": 19, "children": [9, 6], "syn": null, "strings": ["boxed"], "cls_maps": []}, ' \
    '   {"id": 20, "children": [12, 8, 5], "syn": null, "strings": ["jar"], "cls_maps": []}, ' \
    '   {"id": 21, "children": [16, 15, 14, 13, 10], "syn": null, "strings": ["loose"], "cls_maps": []} ' \
    ']'

def flat_tree_pascal_str(map_file_json_str):
    return  '[' \
   '{"children": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "id": 0, "strings": ["entity"], "cls_maps": [], "syn": "Synset(\'entity.n.01\')"}, ' \
   '{"children": [], "id": 1, "strings": ["diningtable"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'table.n.01\')"}, ' \
   '{"children": [], "id": 2, "strings": ["person"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'person.n.01\')"}, ' \
   '{"children": [], "id": 3, "strings": ["horse"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'horse.n.01\')"}, ' \
   '{"children": [], "id": 4, "strings": ["sheep"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'sheep.n.01\')"}, ' \
   '{"children": [], "id": 5, "strings": ["cow"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'cow.n.01\')"}, ' \
   '{"children": [], "id": 6, "strings": ["dog"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'dog.n.01\')"}, ' \
   '{"children": [], "id": 7, "strings": ["cat"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'cat.n.01\')"}, ' \
   '{"children": [], "id": 8, "strings": ["bird"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bird.n.01\')"}, ' \
   '{"children": [], "id": 9, "strings": ["pottedplant"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'plant.n.01\')"}, ' \
   '{"children": [], "id": 10, "strings": ["tvmonitor"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'television.n.01\')"}, ' \
   '{"children": [], "id": 11, "strings": ["sofa"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'sofa.n.01\')"}, ' \
   '{"children": [], "id": 12, "strings": ["chair"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'chair.n.01\')"},' \
   '{"children": [], "id": 13, "strings": ["bottle"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bottle.n.01\')"}, ' \
   '{"children": [], "id": 14, "strings": ["train"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'train.n.01\')"}, ' \
   '{"children": [], "id": 15, "strings": ["bus"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bus.n.01\')"}, ' \
   '{"children": [], "id": 16, "strings": ["motorbike"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'minibike.n.01\')"}, ' \
   '{"children": [], "id": 17, "strings": ["car"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'car.n.01\')"}, ' \
   '{"children": [], "id": 18, "strings": ["bicycle"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bicycle.n.01\')"}, ' \
   '{"children": [], "id": 19, "strings": ["boat"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'boat.n.01\')"}, ' \
   '{"children": [], "id": 20, "strings": ["aeroplane"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'airplane.n.01\')"}' \
   ']'

def tree_pascal_str (map_file_json_str):
    return '[\n' \
    '{"children": [29, 25, 21, 13, 10, 9], "id": 0, "strings": ["entity"], "cls_maps": [], "syn": "Synset(\'entity.n.01\')"}, \n' \
    '{"children": [], "id": 1, "strings": ["diningtable"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'table.n.01\')"}, \n' \
    '{"children": [], "id": 2, "strings": ["person"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'person.n.01\')"}, \n' \
    '{"children": [], "id": 3, "strings": ["horse"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'horse.n.01\')"}, \n' \
    '{"children": [], "id": 4, "strings": ["sheep"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'sheep.n.01\')"}, \n' \
    '{"children": [], "id": 5, "strings": ["cow"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'cow.n.01\')"}, \n' \
    '{"children": [], "id": 6, "strings": ["dog"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'dog.n.01\')"}, \n' \
    '{"children": [], "id": 7, "strings": ["cat"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'cat.n.01\')"}, \n' \
    '{"children": [], "id": 8, "strings": ["bird"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bird.n.01\')"}, \n' \
    '{"children": [], "id": 9, "strings": ["pottedplant"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'plant.n.01\')"}, \n' \
    '{"children": [], "id": 10, "strings": ["tvmonitor"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'television.n.01\')"}, \n' \
    '{"children": [], "id": 11, "strings": ["sofa"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'sofa.n.01\')"}, \n' \
    '{"children": [], "id": 12, "strings": ["chair"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'chair.n.01\')"},\n' \
    '{"children": [], "id": 13, "strings": ["bottle"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bottle.n.01\')"}, \n' \
    '{"children": [], "id": 14, "strings": ["train"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'train.n.01\')"}, \n' \
    '{"children": [], "id": 15, "strings": ["bus"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bus.n.01\')"}, \n' \
    '{"children": [], "id": 16, "strings": ["motorbike"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'minibike.n.01\')"}, \n' \
    '{"children": [], "id": 17, "strings": ["car"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'car.n.01\')"}, \n' \
    '{"children": [], "id": 18, "strings": ["bicycle"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'bicycle.n.01\')"}, \n' \
    '{"children": [], "id": 19, "strings": ["boat"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'boat.n.01\')"}, \n' \
    '{"children": [], "id": 20, "strings": ["aeroplane"], "cls_maps": [' + map_file_json_str + '], "syn": "Synset(\'airplane.n.01\')"}, \n' \
    '{"id": 21, "children": [22, 20, 19], "syn": null, "strings": ["vehicle"], "cls_maps": []},\n' \
    '{"id": 22, "children": [24, 23, 14], "syn": null, "strings": ["land_vehicle"], "cls_maps": []},\n' \
    '{"id": 23, "children": [18, 16], "syn": null, "strings": ["two_wheeled"], "cls_maps": []},\n' \
    '{"id": 24, "children": [17, 15], "syn": null, "strings": ["four_wheeled"], "cls_maps": []},\n' \
    '{"id": 25, "children": [26, 8], "syn": null, "strings": ["animal"], "cls_maps": []},\n' \
    '{"id": 26, "children": [28, 27, 2], "syn": null, "strings": ["mammal"], "cls_maps": []},\n' \
    '{"id": 27, "children": [7, 6], "syn": null, "strings": ["domestic"], "cls_maps": []},\n' \
    '{"id": 28, "children": [5, 4, 3], "syn": null, "strings": ["farm"], "cls_maps": []},\n' \
    '{"id": 29, "children": [12, 11, 1], "syn": null, "strings": ["furniture"], "cls_maps": []}\n' \
    ']'

def create_mappings(data_set_path):
    """
    Creates mapping files for the dataset saved at the given path. Currently supported: Grocery
    :param data_set_path:
    :return:
    """
    if not os.path.exists(os.path.join(data_set_path, "class_map.txt")):
        print("Mapping files do not exist yet - creating them now...")
        import utils.annotations.annotations_helper as mappings
        import pdb; pdb.set_trace()
        class_dict = mappings.create_class_dict(data_set_path)
        mappings.create_map_files(data_set_path, class_dict, training_set=True)
        mappings.create_map_files(data_set_path, class_dict, training_set=False)
        print("Created mapping files!")

def get_tree_str(dataSet, hierarchical=True, map_file = None):
    """
    Function returning a string for a Dataset representing a deserialised TreeMap for given Dataset. Supported Datasets
    are Grocery and Pascal VOC. Can create the mappings for the Grocery Dataset automatically if not created so far.
    :param dataSet: either string "Grocery" or string "pascalVoc"
    :param hierarchical: whether or not the represented Tree should represent the classes in a hierarchical fashion or
    not. If not it equals a usual FastRCNN classifier.
    :param map_file: optional, path to a custom classmapfile if desired. Otherwise the default classmap will be used.
    :return: String containing the desired tree in serialized form.
    """
    if dataSet == "Grocery":
        map_file = map_file if map_file is not None else grocery_map_file
        if not os.path.exists(map_file):
            create_mappings(os.path.dirname(map_file))
        map_file_json=dumps(map_file)
        if hierarchical:
            return tree_grocery_str(map_file_json)
        else:
            return flat_tree_grocery_str(map_file_json)
    elif dataSet == "pascalVoc":
        map_file = map_file if map_file is not None else pascal_map_file
        if not os.path.exists(map_file):
            create_mappings(os.path.dirname(map_file))
        map_file_json = dumps(map_file)
        if hierarchical:
            return tree_pascal_str(map_file_json)
        else:
            return flat_tree_pascal_str(map_file_json)
