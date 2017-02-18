#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Parse ht_spec and run hypertune

import argparse
import types
from pkg.evaluator import Evaluator, EvaluaterQuadratic
from pkg.core.hypertune import *

def key_exists_true(dict, key):
    return key in dict and dict[key]

def split_dict(raw_dict):
    key_val_pairs = []
    for key in raw_dict.keys():
        key_val_pairs.append({key: raw_dict[key]})
    return key_val_pairs

def validate_hyper_param_group(group):
    keylist = group.keys()
    for key in keylist:
        [hp_type, hp_range, hp_distri] = group[key]

        if hp_type == 'Continuous' or hp_type == 'Discrete':
            if len(hp_range) < 2 or len(hp_range) > 3:
                raise ValueError("The value range must be in [low, high, (opt)resolution] format, i.e., "
                                 "2 or 3 values.")
            if hp_range[0] > hp_range[1]:
                raise ValueError("The value range must be in [low, high, (opt)resolution] format, i.e., "
                                 "lower value first.")
            if len(hp_range) == 3:
                if hp_distri == 'Uniform' and hp_range[2] <= 0:
                    raise ValueError("The resolution (or step size) must be larger than 0 for Uniform distribution.")
                elif hp_distri == 'LogUniform' and hp_range[2] <= 1:
                    raise ValueError("The resolution (or step size) must be larger than 1 for LogUniform distribution.")
        elif hp_type == 'Categorical' or hp_type == 'Enumerate':
            if hp_distri != 'Uniform':
                raise ValueError("Only Uniform distribution is supported for Categorical and Enumerate types.")
        else:
            raise ValueError("Unsupported Hyperparameter Type " + hp_type + ".")


def validate_hyper_param_spec(hyper_params):
    if type(hyper_params) is list:
        for group in hyper_params:
            validate_hyper_param_group(group)
    else:
        validate_hyper_param_group(hyper_params)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Hyper-Parameter Search.')

    arg_parser.add_argument('-e', '--cntkexe', dest='cntk_exe', required=True,
                           help='the full path to the CNTK executable')
    arg_parser.add_argument('-p', '--hostfile', dest='host_file',
                           help='the file that contains a list of hosts to run the MPI job. cannot be used together '
                                'with -n.')
    arg_parser.add_argument('-n', '--numproc', dest='numproc', type=int,
                           help='the number of parallel jobs running on the same computer. cannot be used together '
                                'with -p.')
    arg_parser.add_argument('-c', '--cntkcfgfile', dest='cntk_cfg_File', required=True,
                           help='the full path to the CNTK config file')
    arg_parser.add_argument('-s', '--specfile', dest='ht_spec_file',  required=True,
                           help='the file that specifies hyper-parameter search configurations')
    arg_parser.add_argument('-a', '--addition-cntk-arg', dest='addition_cntk_arg',
                           help='the file that specifies hyper-parameter search configurations')
    arg_parser.add_argument('-m', '--modeldir', dest='model_dir', required=True,
                           help='the absolute directory where the CNTK model files will be generated')
    arg_parser.add_argument('-l', '--logdir', dest='log_dir', required=True,
                           help='the absolute directory where the logs will be saved')
    arg_parser.add_argument('-r', '--restart', dest='restart', type=bool, default=False,
                           help='if False (default) it will continue from the checkpoint. Otherwise it will start '
                                'from scratch.')

    args = arg_parser.parse_args()

    if args.host_file is not None and args.numproc is not None:
        raise ValueError("you can set either the host_file (-p) or the number of processes (-n) but not both.")

    with open(args.ht_spec_file, 'r') as f:
        ht_spec = eval(f.read())

    if not 'HyperParams' in ht_spec:
        raise ValueError("HyperParams section does not exist in the hyper-parameter specification file.")

    hyper_params = ht_spec['HyperParams']
    validate_hyper_param_spec(hyper_params)

    if key_exists_true(ht_spec, 'Test'):
        ht_evaluator = EvaluaterQuadratic()
    else:
        if not 'ErrName' in ht_spec:
            raise ValueError("ErrName does not exist in the hyper-parameter specification file.")
        err_node_name = ht_spec['ErrName']

        if not 'ErrWhenFailed' in ht_spec:
            raise ValueError("ErrWhenFailed does not exist in the hyper-parameter specification file.")
        err_when_failed = ht_spec['ErrWhenFailed']

        ht_evaluator = Evaluator(args.cntk_exe, args.host_file, args.numproc, args.cntk_cfg_File, args.addition_cntk_arg,
                                 args.model_dir,
                                 args.log_dir, err_node_name, err_when_failed)

    if key_exists_true(ht_spec, 'CustomizedGroup'):
        if not key_exists_true(ht_spec, 'Additive'):
            raise ValueError("Additive must be true when CustomizedGroup is true")
        if type(hyper_params) is not list:
            raise ValueError("When CustomizedGroup is true hyper parameters must be organized as a list of "
                             "dictionaries")
    else:
        if isinstance(hyper_params, types.ListType):
            print ("WARNING: When CustomizedGroup is false hyper parameters should be organized as a dictionary. "
                   "Converted.")
            hps = dict()
            for group in hyper_params:
                hps.update(group)
            hyper_params = hps

    if ht_spec['GP']:
        if key_exists_true(ht_spec, 'Additive'):
            recommender = GPAddRecommender
            if not key_exists_true(ht_spec, 'CustomizedGroup'):
                hyper_params = split_dict(hyper_params)
        else:
            recommender = GPRecommender
    else:
        recommender = RandRecommender

    if key_exists_true(ht_spec, 'Early'):
        tuner = HyperTuneEarly(hyper_params, ht_evaluator, recommender, ht_spec, args.model_dir, args.restart)
    else:
        tuner = HyperTune(hyper_params, ht_evaluator, recommender, ht_spec, args.model_dir, args.restart)

    tuner.find_best()

    #tuner.clean()

