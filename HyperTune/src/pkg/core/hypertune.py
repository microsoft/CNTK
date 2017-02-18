#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Hypertune module provide two class:
# 1) HyperTune: sequential prediction without early stopping
# 2) HyperTuneEarly: Early stopping (successive halving)


import os
import sys
import time
import types
import json
import os.path
import numpy as np
from .recommender import *


class HyperTune:
    # require 
    # class evaluator:
    #     def evaluate(self, hyper_param_spec, initModel, job_id, training_epochs_per_eval = -1):
    #         return (value, model)
    #     def clean(self):
    

    def __init__(self, hyper_param_spec, evaluator, recommender_type, opt_model_spec, model_dir, restart):
        self.HyperDict = hyper_param_spec
        self.evaluator = evaluator
        self.Recommender = recommender_type(self.HyperDict, opt_model_spec)
        self.check_point_file = os.path.join(model_dir, 'HTCheckpoint.json')

        self.next_trial = 0
        self.read_opt_model_spec(opt_model_spec)

        if restart:
            self.clean()


    def find_best(self):
        self.search()

        self.index_best = np.argmin([result[1] for result in self.candidate_set])
        (self.best_hyper_param, best_error, best_epoch, exError, exVar, job_id, epoch_finished) = self.candidate_set[self.index_best]

        self.print_best(self.best_hyper_param, best_error, best_epoch, job_id)
        return self.best_hyper_param

    def evaluate_job(self, best_error, best_epoch, next_hyper_param, job_id, start_epoch, end_epoch, result_set):
        hyper_param_as_dict = self.Recommender.convert_to_single_dict(next_hyper_param)
        (error, epoch) = self.evaluator.evaluate(hyper_param_as_dict, job_id, start_epoch, end_epoch)

        if best_error is None or error < best_error:
            best_error = error
            best_epoch = epoch

        result_set.append([next_hyper_param, best_error, best_epoch, error, 0, job_id, end_epoch])

        self.print_progress(next_hyper_param, best_error, best_epoch, error, job_id, end_epoch)
        return result_set

    def read_opt_model_spec(self, opt_model_spec):
        #Default value
        # self.max_eval_trials = 50
        # self.training_epochs_per_eval = -1

        if 'HTMaxEvalTrials' in opt_model_spec:
            self.max_eval_trials = opt_model_spec['HTMaxEvalTrials']
        if 'HTTrainingEpochsPerEval' in opt_model_spec:
            self.training_epochs_per_eval = opt_model_spec['HTTrainingEpochsPerEval']
        if 'HTRandomSeed' in opt_model_spec:
            random.seed(opt_model_spec['HTRandomSeed'])
        else:
            random.seed(1)

        return 0

    #used to persist randomstate
    def list_to_tuple(self, v):
        t = ()
        for k in v:
            if type(k) is list:
                t = t + tuple([tuple(k)])
            else:
                t = t + tuple([k])
        return t

    def search(self):
        self.t_start = time.time()
        self.job_id = 1
        self.existing_results = []

        if os.path.isfile(self.check_point_file):
            (self.next_trial, self.job_id, self.existing_results, random_state) = self.load_checkpoint()
            random.setstate(self.list_to_tuple(random_state))

        already_suggested = []
        for i in range(self.next_trial, self.max_eval_trials):
            next_hyper_param = self.Recommender.next_hyper_param(self.existing_results, already_suggested)

            if not self.has_hp_evaluated(next_hyper_param, self.existing_results):
                self.existing_results = self.evaluate_job(None, None, next_hyper_param, self.job_id, 0,
                                                          self.training_epochs_per_eval, self.existing_results)
                self.job_id += 1
            else:
                already_suggested.append(next_hyper_param)
                self.print_hp_evaluated(next_hyper_param)

            self.save_checkpoint(i+1)

        self.candidate_set = self.existing_results


    def delete_checkpoint(self):
        if os.path.isfile(self.check_point_file):
            os.remove(self.check_point_file)

    def load_checkpoint(self):
        f = open(self.check_point_file, 'r')
        data = json.load(f)
        f.close()
        return data

    def save_checkpoint(self, i):
        random_state = random.getstate()
        f = open(self.check_point_file, 'w')
        json.dump([i, self.job_id, self.existing_results, random_state], f, sort_keys=True, indent=4)
        f.close()

    def print_progress(self, hyper_param, best_error, best_epoch, error, job_id, end_epoch):
        t_cur = time.time()
        print(job_id, " Best VErr:", best_error, " at epoch ", best_epoch, \
            " VErr:", error, " After ", end_epoch, " epochs"\
            " T:", round(t_cur - self.t_start, 1), " HP:", hyper_param)

    def print_hp_evaluated(self, hyper_param):
        print("INFO: Recommended Hyper-Param Already Evaluated: ", hyper_param)

    def print_best(self, hyper_param, best_error, best_epoch, job_id):
        t_cur = time.time()
        print("=====Final Test=====")
        print("Best Hyperparameter:  ", hyper_param)
        print("Best Validation Error:  ", best_error, " at epoch ", best_epoch, " at job ", job_id)
        print("Total Time Passed:  ", t_cur - self.t_start)


    def clean(self):
        self.delete_checkpoint()
        self.evaluator.clean()

    # existing_results is list of(hyper_param, best_error, best_epoch, exError, exVar, job_id, epoch_finished)
    def has_hp_evaluated(self, new_hp, existing_results):
        hp_evaluated = False
        for result in existing_results:
            if new_hp == result[0]:
                hp_evaluated = True
                break

        return hp_evaluated

    # existing_results is list of(hyper_param, best_error, best_epoch, exError, exVar, job_id, epoch_finished)
    def get_job_info(self, new_hp, existing_results):
        for result in existing_results:
            if new_hp == result[0]:
                return result

        return None, None, None, None, None, None, 0

class HyperTuneEarly(HyperTune):

    def read_opt_model_spec(self, opt_model_spec):
        #Default value
        self.initial_eval_trials = 100
        self.max_round = 4
        self.training_epochs_per_eval = [2, 4, 8, 16]
        self.suggest_rate = [0.5, 0.25, 0.2, 0.15]
        self.next_rate = 0.5
        self.calibration = True
        self.calibration_ratio_mean = 0.6
        self.calibration_ratio_var = 0.6

        if 'HTInitialEvalTrials' in opt_model_spec:
            self.initial_eval_trials = opt_model_spec['HTInitialEvalTrials']
        if 'HTMaxRound' in opt_model_spec:
            self.max_round = opt_model_spec['HTMaxRound']
        if 'HTTrainingEpochsPerEval' in opt_model_spec:
            self.training_epochs_per_eval = opt_model_spec['HTTrainingEpochsPerEval']
        if 'HTSuggestRate' in opt_model_spec:
            self.suggest_rate = opt_model_spec['HTSuggestRate']
        if 'HTNextRate' in opt_model_spec:
            self.next_rate = opt_model_spec['HTNextRate']
        if 'HTCalibrate' in opt_model_spec:
            self.calibration = opt_model_spec['HTCalibrate']
        if 'HTCalibrateRatio_M' in opt_model_spec:
            self.calibration_ratio_mean = opt_model_spec['HTCalibrateRatio_M']
        if 'HTCalibrateRatio_V' in opt_model_spec:
            self.calibration_ratio_var = opt_model_spec['HTCalibrateRatio_V']
        if 'HTRandomSeed' in opt_model_spec:
            random.seed(opt_model_spec['HTRandomSeed'])
        else:
            random.seed(1)

        if self.max_round > len(self.training_epochs_per_eval):
            raise ValueError("HTMaxRound must be no larger than the number of entries in HTTrainingEpochsPerEval.")
        if self.max_round ==0:
            raise ValueError("HTMaxRound must be larger than 0.")

        if len(self.training_epochs_per_eval) < self.max_round:
            raise ValueError("Number of values in TrainingEpochsPerEval cannot be less than HTMaxRound.")

        for i in range(1, self.max_round):
            if self.training_epochs_per_eval[i] <= self.training_epochs_per_eval[i-1]:
                raise ValueError("In TrainingEpochsPerEval each later value must be larger than the earlier value.")

        return 0


    #result sets are in format of (hyper_param, best_error, best_epoch, exError, exVar, epoch_finished, job_id)
    def search(self):
        self.t_start = time.time()
        self.job_id = 1
        start_round = 0
        next_trial_c = 0
        next_trial_s = 0

        if os.path.isfile(self.check_point_file):
            (start_round, next_trial_c, next_trial_s, self.job_id, self.candidate_set, \
                self.eliminated_set_calib, self.newcandidate_set, random_state) = self.load_checkpoint()
            random.setstate(self.list_to_tuple(random_state))

        for cur_round in range(start_round, self.max_round):
            end_epoch = self.training_epochs_per_eval[cur_round]

            # Calculating how many by competing and how many by suggesting
            total_size = int(round(self.initial_eval_trials * (self.next_rate ** cur_round)))
            suggest_size = int(round(total_size * self.suggest_rate[cur_round]))
            cand_size = total_size - suggest_size

            # Initializing / Ruling out bad performing parameters in Candidate
            if next_trial_c == 0:
                if cur_round == 0:
                    self.candidate_set = []
                    for i in range(cand_size):
                        next_hyper_param = self.Recommender.sample()
                        if not self.has_hp_evaluated(next_hyper_param, self.candidate_set):
                            # (hyper_param, bestError, bestEpoch, exError, exVar, job_id, epoch_finished)
                            self.candidate_set.append((next_hyper_param, None, 0, 1, 0, None, 0))
                    self.eliminated_set_calib = []
                else:
                    self.candidate_set.sort(key = lambda tup: tup[1])  #based on best_error
                    self.eliminated_set_calib.extend(self.candidate_set[cand_size:])
                    self.candidate_set = self.candidate_set[:cand_size]
                self.newcandidate_set = []

            cand_size = len(self.candidate_set)
            # Run existing parameters for additional rounds
            print("=====candidate_set Performance=====")
            for i in range(next_trial_c, cand_size):
                (next_hyper_param, best_error, best_epoch, exError, exVar, old_job_id, epoch_finished) = \
                    self.candidate_set[i]

                if old_job_id is None:
                    use_job_id = self.job_id
                    self.job_id = self.job_id + 1
                else:
                    use_job_id = old_job_id

                self.newcandidate_set = self.evaluate_job(best_error, best_epoch, next_hyper_param, use_job_id,
                                                          epoch_finished, end_epoch, self.newcandidate_set)

                self.save_checkpoint(cur_round, i+1, 0)

            # Calibrate eliminated Set
            if next_trial_s == 0 and self.calibration:
                self.calibrate()

                
            # Run recommended parameters for full rounds
            print("=====Recommended settings Performance=====")
            already_suggested = []
            for j in range(next_trial_s, suggest_size):
                existing_results = self.eliminated_set_calib + self.newcandidate_set
                next_hyper_param = self.Recommender.next_hyper_param(existing_results, already_suggested)
                if not self.has_hp_evaluated(next_hyper_param, self.newcandidate_set):
                    (hyper_param, best_error, best_epoch, exError, exVar, use_job_id, start_epoch) = \
                    self.get_job_info(next_hyper_param, self.eliminated_set_calib)

                    if use_job_id is None:
                        use_job_id = self.job_id
                        self.job_id = self.job_id + 1
                        start_epoch = 0

                    self.newcandidate_set = self.evaluate_job(best_error, best_epoch, next_hyper_param, use_job_id,
                                                                start_epoch, end_epoch, self.newcandidate_set)
                else:
                    already_suggested.append(next_hyper_param)
                    self.print_hp_evaluated(next_hyper_param)

                self.save_checkpoint(cur_round, cand_size, j+1)

            self.candidate_set = self.newcandidate_set
            next_trial_c = 0
            next_trial_s = 0

        self.training_epochs_per_eval = end_epoch


    def save_checkpoint(self, cur_round, i, j):
        random_state = random.getstate()
        f = open(self.check_point_file, 'w')
        json.dump([cur_round, i, j, self.job_id, self.candidate_set, \
            self.eliminated_set_calib, self.newcandidate_set, random_state], f, sort_keys=True, indent=4)
        f.close()


    def calibrate(self):
        # based on newcandidate_set, candidate_set to change eliminated_set_calib
        oldErr = np.array([result[3] for result in self.candidate_set])
        newErr = np.array([result[3] for result in self.newcandidate_set])
        dif = newErr - oldErr
        mean = np.mean(dif)
        var = np.var(dif)
        ratio_mean = self.calibration_ratio_mean
        ratio_var = self.calibration_ratio_var

        for i in range(len(self.eliminated_set_calib)):
            (hp, best_error, best_epoch, error, noiseVar, job_id, finished_epoch) = self.eliminated_set_calib[i]
            self.eliminated_set_calib[i] = (hp, best_error, best_epoch, error + mean * ratio_mean, \
                noiseVar + var * ratio_var, job_id, finished_epoch)


if __name__ == '__main__':
    cur_evaluator = Evaluator('cntk_exe', 'host_file', 1, 'cntk_cfg_File', 'model_dir', 'log_dir', 'err_node_name')
    cur_hopt = HyperTune('src/Examples/Cifar/CifarAdd.py', cur_evaluator)
    cur_hopt.setRecommender(GPAddRecommender)
    cur_hopt.find_best()
    cur_hopt.clean()
    # cur_evaluator = evaluator('Examples/Simple_win.cntk')
    # cur_hopt = HyperTune('Examples/Simple_hyper.py', cur_evaluator)
    # cur_hopt.setRecommender(GaussianProcess)
    # cur_hopt.search()
    # hp = cur_hopt.getBestHyperPara()
    # cur_hopt.clean()
