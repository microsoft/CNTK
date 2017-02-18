#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# interface with CNTK
# class evaluator:
#     def evaluate(self, hyper_param_instance, continue_job_id, job_id, numIter = -1):


import re
import os
import subprocess
import numpy as np
from shutil import copyfile, rmtree


class LogParser:
    # For parsing the output log file
    def __init__(self, log_file, err_node_name, err_when_failed):
        self.log_file = log_file
        self.err_node_name = err_node_name
        self.err_when_failed = err_when_failed

    def get__criterion_core(self, dataTag):
        data_set = dataTag
        error_search = re.compile(
            'Finished Epoch\[\s*?(\d*) of \s*(\d*)\]: ' + 
            '\[' + data_set + '\].*' + self.err_node_name + ' = (\d*\.?\d*)')

        epoch_id = 0
        criterion = float('inf')

        best_error = None
        best_epoch = 0

        f = open(self.log_file, 'r')
        for line in f:
            results = error_search.search(line)
            if results:
                epoch_id = float(results.group(1))
                total_epochs = float(results.group(2))
                criterion = float(results.group(3))
                if best_error is None or best_error > criterion:
                    best_error = criterion
                    best_epoch = epoch_id
        return best_error, best_epoch, f, criterion

    def get_criterion(self):
        data_set = 'Validate'
        best_error, best_epoch, f, criterion = self.get__criterion_core(data_set)
        if criterion == float('inf'):
            data_set = 'Training'
            best_error, best_epoch, f, criterion = self.get__criterion_core(data_set)

        exception_search = re.compile('EXCEPTION occurred: (.*)')

        # if cannot find the value (may be incorrect setup or cntk crashed with NAN)
        if criterion == float('inf'):
            exception_occurred = False
            exception_msg = None
            f.seek(0, 0)
            for line in f:
                results = exception_search.search(line)
                if results:
                    exception_occurred = True
                    exception_msg = results.group(1)

            if exception_occurred:
                print('WARNING: Exception occurred: ' + exception_msg)
            else:
                print('WARNING: Cannot find \'' + data_set + '\' or \'' + self.err_node_name + '\' in the log file. Check the log to find reason.')

            best_error = self.err_when_failed
            best_epoch = 0

        f.close()

        return best_error, best_epoch


class Evaluator:
    # For evaluating current hyperparameter on CNTK
    def __init__(self, cntk_exe, host_file, numproc, cntk_cfg_File, addition_cntk_arg, model_dir, log_dir,
                 err_node_name, err_when_failed):
        """
        :type cntk_exe: basestring
        :type cntk_cfg_File: basestring
        :type host_file: basestring
        :type numproc: int
        :type model_dir: basestring
        :type log_dir: basestring
        :type err_node_name: basestring
        :type err_when_failed: float
        """
        self.cntk_exe = cntk_exe
        self.host_file = host_file
        self.numproc = numproc
        self.cntk_cfg_File = cntk_cfg_File
        self.addition_cntk_arg = addition_cntk_arg
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.err_node_name = err_node_name
        self.err_when_failed = err_when_failed

        if self.host_file is not None or (self.numproc is not None and self.numproc > 1):
            self.parallel_mode = True
        else:
            self.parallel_mode = False

    def evaluate(self, hyper_param_instance, job_id, start_epoch, end_epoch):
        self.run_experiment(hyper_param_instance, job_id, start_epoch, end_epoch)
        return self.collect_result(job_id)

    def run_experiment(self, hyper_param_instance, job_id, start_epoch, end_epoch):
        # put temp model and check point file under model_dir
        # put log file under log_dir

        # construct cmd and call
        cmd_line = []
        if self.host_file is not None or self.numproc is not None:
            cmd_line =['mpiexec']
            if self.host_file is not None:
                cmd_line.append('-hostfile')
                cmd_line.append(self.host_file)
            else:
                cmd_line.append('-n')
                cmd_line.append(str(self.numproc))

        cmd_line.append(self.cntk_exe)
        cmd_line.append('configFile=' + self.cntk_cfg_File)

        full_model_path = self._get_model_path(job_id)
        full_log_prefix = self._get_log_prefix(job_id)
        
        if start_epoch > 0:
            start_model = full_model_path + "." + str(start_epoch)
            self.rename_file_if_exists(full_model_path, start_model)
            self.rename_file_if_exists(full_model_path + ".ckp", start_model + ".ckp")

        cmd_line.append('modelPath=' + full_model_path)
        cmd_line.append('stderr=' + full_log_prefix)
        
        if self.parallel_mode:
            cmd_line.append('parallelTrain=true')
        else:
            cmd_line.append('parallelTrain=false')

        if self.addition_cntk_arg is not None:
            cmd_line.append(self.addition_cntk_arg)

        for key in hyper_param_instance.keys():
            cmd_line.append(key + '=' + str(hyper_param_instance[key]))

        # Control Number of iterations
        cmd_line.append('maxEpochs=' + str(end_epoch))
        
        print(cmd_line)
        subprocess.call(cmd_line)
        return

    def rename_file_if_exists(self, from_file, to_file):
        if os.path.isfile(from_file):
            if os.path.isfile(to_file):
                print("WARNING: trying to rename " + from_file + " to existing file " + to_file)
                os.remove(to_file)
            os.rename(from_file, to_file)
        else:
            print("WARNING: trying to rename " + from_file + " (does not exist) to " + to_file)

    def collect_result(self, job_id):
        # full log file is a combination of log prefix and command section
        # the new version may allow using whatever passed in
        full_log_prefix = self._get_log_prefix(job_id)
        log_ext = '.log'
        log_rank = ''
#        if self.parallel_mode:
#            log_rank = 'rank0'

        log_file_name = full_log_prefix + log_rank

        if not os.path.isfile(log_file_name):
            command_to_run = ""
            f = open(self.cntk_cfg_File, 'r')
            command_search = re.compile('^\s*command\s*=\s*([^#\s]+)')
            for line in f:
                results = command_search.search(line)
                if results:
                    command_to_run = results.group(1).replace(':', '_')
                    break

            if command_to_run == "":
                raise ValueError("cannot find 'command=' in the cntk config file " + self.cntk_cfg_File)

            log_file_name = full_log_prefix + '_' + command_to_run + log_ext + log_rank

        log_parser = LogParser(log_file_name, self.err_node_name, self.err_when_failed)
        return log_parser.get_criterion()

    def delete_model_for_job(self, job_id):
        rmtree(self._get_model_path(job_id))

    def delete_all_models(self):
        rmtree(self.model_dir)

    def delete_all_logs(self):
        rmtree(self.log_dir)

    def clean(self):
        self.delete_all_models()
        self.delete_all_logs()

    def _get_model_path(self, job_id):
        model_dir_for_job = os.path.join(self.model_dir, str(job_id))
        if not os.path.exists(model_dir_for_job):
            os.makedirs(model_dir_for_job)

        model_name = 'model.dnn'
        full_model_path = os.path.join(model_dir_for_job, model_name)
        return full_model_path

    def _get_log_prefix(self, job_id):
        log_dir_for_job = os.path.join(self.log_dir, str(job_id))
        if not os.path.exists(log_dir_for_job):
            os.makedirs(log_dir_for_job)

        log_prefix = os.path.join(log_dir_for_job, "tune.log")
        return log_prefix

# For test purpose
class EvaluaterQuadratic:
    # For test evaluating on simple quadratic function
    def evaluate(self, hyper_param_instance, continue_job_id, job_id, numIter = -1):
        tempX = hyper_param_instance
        # return ((tempX['x1'] + 0.3)**2 + (np.log(tempX['x2']) - 2)**2 \
        #     + 3*(tempX['x3'] - 0.8)**2 + tempX['x4'], None)

        #return ((tempX['x1'] + np.log(tempX['x2']) - 1.7)**2 + (np.log(tempX['x2']) - 2)**2 \
        #    + 3*(tempX['x3'] - 0.8)**2 + tempX['x4'], None)
        # return ((tempX['x1'] + 0.3)**2 + (np.log(tempX['x2']) - 2)**2, None)
        # return ((tempX['x1'] + np.log(tempX['x2']) - 1.7)**2 + (np.log(tempX['x2']) - 2)**2 \
        #     + 3*(tempX['x3'] - 0.8)**2 + tempX['x4'] + (tempX['x5'] + 0.7)**2 \
        #     + (tempX['x6'] - 0.2)**2 + (tempX['x7'] - 0.6)**2 \
        #     + (tempX['x8'] + 0.4)**2 + (tempX['x9'] - 0.5)**2 + 0.1*tempX['x8']*tempX['x9']\
        #     , None)
        return ((tempX['x1'] + 0.3)**2 + (np.log(tempX['x2']) - 2)**2 \
             + 3*(tempX['x3'] - 0.8)**2 + tempX['x4'] + (tempX['x5'] + 0.7)**2 \
             + (tempX['x6'] - 0.2)**2 + (tempX['x7'] - 0.6)**2 \
             + (tempX['x8'] + 0.4)**2 + (tempX['x9'] - 0.5)**2 \
             , None)
        # optimal is (-0.3, 7.39, 1, 1)

    def clean(self):
        return 0

    def testError(self):
        return 1.12

