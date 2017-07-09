# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.contrib.model2cntk.utils import format


class GlobalConf(object):
    def __init__(self):
        # base terms
        self.source_solver = SourceSolverConf()
        # the solver of script converter
        self.script_solver = ScriptSolverConf()
        # the solver of weights converter
        self.model_solver = ModelSolverConf()
        # the solver of valid
        self.valid_solver = ValidConf()
        # the solver of evluation
        self.classify_eval_solver = ClassifyEvalConf()


class SourceSolverConf(object):
    def __init__(self):
        self.source = None              # the source of script and model, Caffe or TensorFlow. For now, just Caffe
        self.model_path = None,         # the path of model
        self.weights_path = None        # the path of script
        self.tensor = None              # dict of inputs
        self.phase = None               # the phase to be converted, 1 for test, 0 for training


class ScriptSolverConf(object):
    def __init__(self):
        pass


class ModelSolverConf(object):
    def __init__(self):
        self.cntk_model_path = None     # the path of exported cntk model
        self.cntk_tensor = None         # in case, need to reshape the data


class ValidConf(object):
    def __init__(self):
        self.save_path = None
        self.val_inputs = []
        self.val_nodes = []


class ClassifyEvalConf(object):
    def __init__(self):
        self.top_n = 1
        self.label_tensor = None
        self.index_map = None           # the index map of tested dataset
        self.mean_file = None           # the mean file of tested dataset
        self.dataset_size = 0
        self.batch_size = 1
        self.crop_ratio = None
        self.crop_type = None


class SegmentEvalConf(object):
    def __init__(self):
        pass


class DetectEvalConf(object):
    def __int__(self):
        pass


DICT_CONFIG_CLASSES = (GlobalConf, SourceSolverConf, ScriptSolverConf, ModelSolverConf, ValidConf, ClassifyEvalConf,
                       SegmentEvalConf, DetectEvalConf)


def load_sub_conf(meta_dict, target_type):
    class_conf = target_type()
    for key, value in meta_dict.items():
        camel_key = format.camel_to_snake(key)
        if camel_key not in dir(class_conf):
            continue
        raw_attr = getattr(class_conf, camel_key)
        if isinstance(raw_attr, DICT_CONFIG_CLASSES):
            setattr(class_conf, camel_key, load_sub_conf(value, type(raw_attr)))
        else:
            setattr(class_conf, camel_key, value)
    return class_conf


def load_conf(path):
    json_attributes = format.json_parser(path)
    global_conf = load_sub_conf(json_attributes, GlobalConf)
    return global_conf
