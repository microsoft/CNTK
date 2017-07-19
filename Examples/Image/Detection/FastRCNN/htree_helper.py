# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os

my_path = os.path.dirname(os.path.realpath(__file__))
my_path = my_path.replace("\\","/")
grocery_map_file = my_path + "/../../DataSets/Grocery/Class_map.txt"
pascal_map_file = my_path + "/../../DataSets/Pascal/class_map.txt"

def flat_tree_grocery_str(map_file):
    return '[' \
    '   {"id": 0, "childrens": [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "syn": "Synset(\'physical_entity.n.01\')", "strings": ["physical_entity.n.01"], "cls_maps": []}, ' \
    '   {"id": 1, "childrens": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 2, "childrens": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 3, "childrens": [], "syn": "Synset(\'juice.n.01\')", "strings": ["orangeJuice"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 4, "childrens": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 5, "childrens": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gerkin"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 6, "childrens": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggBox"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 7, "childrens": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 8, "childrens": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["joghurt"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 9, "childrens": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 10, "childrens": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 11, "childrens": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 12, "childrens": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 13, "childrens": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 14, "childrens": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 15, "childrens": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 16, "childrens": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": ["' + map_file + '"]}' \
    ']'

def tree_grocery_str(map_file):
    return '[' \
    '   {"id": 0, "childrens": [21, 17], "syn": "Synset(\'physical_entity.n.01\')", "strings": ["physical_entity.n.01"], "cls_maps": []}, ' \
    '   {"id": 1, "childrens": [], "syn": "Synset(\'water.n.01\')", "strings": ["water"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 2, "childrens": [], "syn": "Synset(\'champagne.n.01\')", "strings": ["champagne"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 3, "childrens": [], "syn": "Synset(\'juice.n.01\')", "strings": ["orangeJuice"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 4, "childrens": [], "syn": "Synset(\'catsup.n.01\')", "strings": ["ketchup"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 5, "childrens": [], "syn": "Synset(\'gherkin.n.01\')", "strings": ["gerkin"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 6, "childrens": [], "syn": "Synset(\'egg.n.02\')", "strings": ["eggBox"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 7, "childrens": [], "syn": "Synset(\'milk.n.01\')", "strings": ["milk"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 8, "childrens": [], "syn": "Synset(\'yogurt.n.01\')", "strings": ["joghurt"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 9, "childrens": [], "syn": "Synset(\'butter.n.01\')", "strings": ["butter"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 10, "childrens": [], "syn": "Synset(\'tomato.n.01\')", "strings": ["tomato"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 11, "childrens": [], "syn": "Synset(\'tabasco.n.01\')", "strings": ["tabasco"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 12, "childrens": [], "syn": "Synset(\'mustard.n.01\')", "strings": ["mustard"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 13, "childrens": [], "syn": "Synset(\'pepper.n.01\')", "strings": ["pepper"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 14, "childrens": [], "syn": "Synset(\'onion.n.01\')", "strings": ["onion"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 15, "childrens": [], "syn": "Synset(\'orange.n.01\')", "strings": ["orange"], "cls_maps": ["' + map_file + '"]}, ' \
    '   {"id": 16, "childrens": [], "syn": "Synset(\'avocado.n.01\')", "strings": ["avocado"], "cls_maps": ["' + map_file + '"]},' \
    '   {"id": 17, "childrens": [20, 19, 18], "syn": null, "strings": ["packaged"], "cls_maps": []}, ' \
    '   {"id": 18, "childrens": [11, 7, 4, 3, 2, 1], "syn": null, "strings": ["bottled"], "cls_maps": []}, ' \
    '   {"id": 19, "childrens": [9, 6], "syn": null, "strings": ["boxed"], "cls_maps": []}, ' \
    '   {"id": 20, "childrens": [12, 8, 5], "syn": null, "strings": ["jar"], "cls_maps": []}, ' \
    '   {"id": 21, "childrens": [16, 15, 14, 13, 10], "syn": null, "strings": ["loose"], "cls_maps": []} ' \
    ']'

def flat_tree_pascal_str(map_file):
    return  '[' \
   '{"childrens": [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], "id": 0, "strings": ["entity"], "cls_maps": [], "syn": "Synset(\'entity.n.01\')"}, ' \
   '{"childrens": [], "id": 1, "strings": ["diningtable"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'table.n.01\')"}, ' \
   '{"childrens": [], "id": 2, "strings": ["person"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'person.n.01\')"}, ' \
   '{"childrens": [], "id": 3, "strings": ["horse"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'horse.n.01\')"}, ' \
   '{"childrens": [], "id": 4, "strings": ["sheep"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'sheep.n.01\')"}, ' \
   '{"childrens": [], "id": 5, "strings": ["cow"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'cow.n.01\')"}, ' \
   '{"childrens": [], "id": 6, "strings": ["dog"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'dog.n.01\')"}, ' \
   '{"childrens": [], "id": 7, "strings": ["cat"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'cat.n.01\')"}, ' \
   '{"childrens": [], "id": 8, "strings": ["bird"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bird.n.01\')"}, ' \
   '{"childrens": [], "id": 9, "strings": ["pottedplant"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'plant.n.01\')"}, ' \
   '{"childrens": [], "id": 10, "strings": ["tvmonitor"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'television.n.01\')"}, ' \
   '{"childrens": [], "id": 11, "strings": ["sofa"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'sofa.n.01\')"}, ' \
   '{"childrens": [], "id": 12, "strings": ["chair"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'chair.n.01\')"},' \
   '{"childrens": [], "id": 13, "strings": ["bottle"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bottle.n.01\')"}, ' \
   '{"childrens": [], "id": 14, "strings": ["train"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'train.n.01\')"}, ' \
   '{"childrens": [], "id": 15, "strings": ["bus"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bus.n.01\')"}, ' \
   '{"childrens": [], "id": 16, "strings": ["motorbike"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'minibike.n.01\')"}, ' \
   '{"childrens": [], "id": 17, "strings": ["car"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'car.n.01\')"}, ' \
   '{"childrens": [], "id": 18, "strings": ["bicycle"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bicycle.n.01\')"}, ' \
   '{"childrens": [], "id": 19, "strings": ["boat"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'boat.n.01\')"}, ' \
   '{"childrens": [], "id": 20, "strings": ["aeroplane"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'airplane.n.01\')"}' \
   ']'

def tree_pascal_str (map_file):
    return '[\n' \
    '{"childrens": [29, 25, 21, 13, 10, 9], "id": 0, "strings": ["entity"], "cls_maps": [], "syn": "Synset(\'entity.n.01\')"}, \n' \
    '{"childrens": [], "id": 1, "strings": ["diningtable"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'table.n.01\')"}, \n' \
    '{"childrens": [], "id": 2, "strings": ["person"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'person.n.01\')"}, \n' \
    '{"childrens": [], "id": 3, "strings": ["horse"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'horse.n.01\')"}, \n' \
    '{"childrens": [], "id": 4, "strings": ["sheep"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'sheep.n.01\')"}, \n' \
    '{"childrens": [], "id": 5, "strings": ["cow"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'cow.n.01\')"}, \n' \
    '{"childrens": [], "id": 6, "strings": ["dog"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'dog.n.01\')"}, \n' \
    '{"childrens": [], "id": 7, "strings": ["cat"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'cat.n.01\')"}, \n' \
    '{"childrens": [], "id": 8, "strings": ["bird"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bird.n.01\')"}, \n' \
    '{"childrens": [], "id": 9, "strings": ["pottedplant"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'plant.n.01\')"}, \n' \
    '{"childrens": [], "id": 10, "strings": ["tvmonitor"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'television.n.01\')"}, \n' \
    '{"childrens": [], "id": 11, "strings": ["sofa"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'sofa.n.01\')"}, \n' \
    '{"childrens": [], "id": 12, "strings": ["chair"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'chair.n.01\')"},\n' \
    '{"childrens": [], "id": 13, "strings": ["bottle"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bottle.n.01\')"}, \n' \
    '{"childrens": [], "id": 14, "strings": ["train"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'train.n.01\')"}, \n' \
    '{"childrens": [], "id": 15, "strings": ["bus"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bus.n.01\')"}, \n' \
    '{"childrens": [], "id": 16, "strings": ["motorbike"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'minibike.n.01\')"}, \n' \
    '{"childrens": [], "id": 17, "strings": ["car"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'car.n.01\')"}, \n' \
    '{"childrens": [], "id": 18, "strings": ["bicycle"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'bicycle.n.01\')"}, \n' \
    '{"childrens": [], "id": 19, "strings": ["boat"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'boat.n.01\')"}, \n' \
    '{"childrens": [], "id": 20, "strings": ["aeroplane"], "cls_maps": ["' + map_file + '"], "syn": "Synset(\'airplane.n.01\')"}, \n' \
    '{"id": 21, "childrens": [22, 20, 19], "syn": null, "strings": ["vehicle"], "cls_maps": []},\n' \
    '{"id": 22, "childrens": [24, 23, 14], "syn": null, "strings": ["land_vehicle"], "cls_maps": []},\n' \
    '{"id": 23, "childrens": [18, 16], "syn": null, "strings": ["two_wheeled"], "cls_maps": []},\n' \
    '{"id": 24, "childrens": [17, 15], "syn": null, "strings": ["four_wheeled"], "cls_maps": []},\n' \
    '{"id": 25, "childrens": [26, 8], "syn": null, "strings": ["animal"], "cls_maps": []},\n' \
    '{"id": 26, "childrens": [28, 27, 2], "syn": null, "strings": ["mammal"], "cls_maps": []},\n' \
    '{"id": 27, "childrens": [7, 6], "syn": null, "strings": ["domestic"], "cls_maps": []},\n' \
    '{"id": 28, "childrens": [5, 4, 3], "syn": null, "strings": ["farm"], "cls_maps": []},\n' \
    '{"id": 29, "childrens": [12, 11, 1], "syn": null, "strings": ["furniture"], "cls_maps": []}\n' \
    ']'

def create_mappings(data_set_path):
    """
    Creates mapping files for the dataset saved at the given path. Currently supported: Grocery
    :param data_set_path:
    :return:
    """
    data_set_path
    if not os.path.exists(os.path.join(data_set_path, "Class_map.txt")):
        print("Mapping files do not exist yet - creating them now...")
        import utils.mappings.map_file_helper as mappings
        class_dict = mappings.create_class_dict(data_set_path)
        mappings.create_map_files(data_set_path, class_dict, training_set=True)
        mappings.create_map_files(data_set_path, class_dict, training_set=False)
        print("Created mapping files!")

def get_tree_str(dataSet, hierarchical=True, map_file = None):
    """
    Function returning a string for a Dataset representing a deserialised TreeMap for given Dataset. Supported Datasets are Grocery and Pascal VOC. Can create the mapings for the Grocery Dataset automatically if not created so far.
    :param dataSet: either string "Grocery" or string "pascalVoc"
    :param hierarchical: whether or not the represented Tree should represent the classes in a hierarchical fashion or not. If not it equals a usual FastRCNN classifier.
    :param map_file: optional, path to a custom classmapfile if desired. Otherwise the default classmap will be used.
    :return:
    """
    if dataSet == "Grocery":
        map_file = map_file if map_file is not None else grocery_map_file
        if not os.path.exists(map_file):
            create_mappings(os.path.dirname(map_file))
        if hierarchical:
            return tree_grocery_str(map_file)
        else:
            return flat_tree_grocery_str(map_file)
    elif dataSet == "pascalVoc":
        map_file = map_file if map_file is not None else pascal_map_file
        if not os.path.exists(map_file):
            create_mappings(os.path.dirname(map_file))
        if hierarchical:
            return tree_pascal_str(map_file)
        else:
            return flat_tree_pascal_str(map_file)
