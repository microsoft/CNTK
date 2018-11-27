# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.contrib.crosstalkcaffe.utils import format


class GlobalConf(object):
    '''
     The definition of global configuration
    '''
    def __init__(self):
        # base terms
        self.source_solver = SourceSolverConf()
        # the solver of script converter
        self.script_solver = ScriptSolverConf()
        # the solver of weights converter
        self.model_solver = ModelSolverConf()
        # the solver of valid
        self.valid_solver = ValidConf()


class SourceSolverConf(object):
    '''
     The source solver
    '''
    def __init__(self):
        self.source = None
        self.model_path = None
        self.weights_path = None
        self.tensor = None
        self.phase = None


class ScriptSolverConf(object):
    '''
     The script solver
    '''
    def __init__(self):
        pass


class ModelSolverConf(object):
    '''
     The model solver
    '''
    def __init__(self):
        self.cntk_model_path = None
        self.cntk_tensor = None


class ValidConf(object):
    '''
     The validation solver
    '''
    def __init__(self):
        self.save_path = None
        self.val_inputs = []
        self.val_nodes = []


DICT_CONFIG_CLASSES = (GlobalConf, SourceSolverConf, ScriptSolverConf, ModelSolverConf, ValidConf)


def _load_sub_conf(meta_dict, target_type):
    class_conf = target_type()
    for key, value in meta_dict.items():
        camel_key = format.camel_to_snake(key)
        if camel_key not in dir(class_conf):
            continue
        raw_attr = getattr(class_conf, camel_key)
        if isinstance(raw_attr, DICT_CONFIG_CLASSES):
            setattr(class_conf, camel_key, _load_sub_conf(value, type(raw_attr)))
        else:
            setattr(class_conf, camel_key, value)
    return class_conf


def load_conf(conf_path):
    '''
     Analysis global configuration file into dict

    Args:
        conf_path (str): the path to global configuration file

    Return:
        :class:`~cntk.contrib.crosstalkcaffe.utils.GlobalConf`
    '''
    json_attributes = format.json_parser(conf_path)
    global_conf = _load_sub_conf(json_attributes, GlobalConf)
    return global_conf
