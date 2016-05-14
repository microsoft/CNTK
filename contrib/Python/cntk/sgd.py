# Copyright (c) Microsoft. All rights reserved.
#Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================



class SGDParams:
    """
    This class encapsulates the training parameters of Stochastic Gradien
    Descent. 

    * Training process control
        * **model_path**: the full path to save the final model. Must be provided and points to a valid file name.
        * **train_criterion_node_name**: the name of the training criterion node. If not provided the default training criterion node in the network will be used.
        * **eval_criterion_node_name**: the name of the evaluation criterion node. If not provided the default evaluation criterion node in the network will be used.
        * **epoch_size**: epoch size, i.e., the number of samples in each epoch. Often it is the dataset size, but can also be different. An intermediate model and other check point information is saved for each epoch. When set to 0 the whole dataset size is used.
        * **keep_check_point_files**: whether you want to keep the check point file after a new epoch starts. Valid values are true and false (default).
        * **max_epochs**: maximum number of epochs to run.
        * **minibatch_size: minibatch size for each epoch. Default value is 256. Can use syntax such as 128*2**:1024 which means using minibatch size 128 for 2 epochs and then 1024 for the rest.
        * **dropout_rate: dropout rate during the training procedure. Default is 0.0. Can use syntax such as 0.5*10**:0.2 which means using dropout rate 0.5 for 10 epochs and then 0.2 for the rest.
        * **max_temp_mem_size_in_samples_for_cnn**: maximum temporary memory used (in number of samples) when packaging and unpackaging input features. Default is 0, which means using any value as needed. Useful to control the memory foot print esp. when run under GPU.
    
    * Learning rate and momentum control
        * **learning_rates_per_mb: learning rates per minibatch. Useful when you want to use the same learning rate while the minibatch size is changed. Can use syntax such as 0.8*10**:0.2 which means using the learning rate 0.8 for 10 epochs and then 0.2 for the rest. learningRatesPerMB may be missing, for example, when learningRatesPerSample is provided or the automatic learning rate determination algorithm is used.
        * **learning_rates_per_sample: learning rates per sample. Useful when you want to keep the learning rates per sample constant, i.e., automatically increases effective learning rate for the minibatch when the minibatch size is increased. Can use syntax such as 0.008*10**:0.002 which means using the learning rate 0.008 for 10 epochs and then 0.002 for the rest. learningRatesPerSample may be missing, for example, when learningRatesPerMB is provided or the automatic learning rate determination algorithm is used.
        * **momentum_per_mb: momentum per minibatch. Default is 0.9. Can use syntax such as 0.1*2**:0.9 which means using momentum 0.1 for 2 epochs and then 0.9 for the rest.
        * **momentum_per_sample: momentum per sample. Useful when you want to keep the momentum per sample constant, i.e., automatically scales effective momentum for the minibatch when the minibatch size is changed. Can use syntax such as 0.9996*10**:0.998 which means using the per sample momentum 0.9996 for 10 epochs and then 0.998 for the rest. momentumPerSample may be missing, for example, when momentumPerMB is provided.
        * **momentum_as_time_constant**: number of samples after which the contribution is decayed to e^-1
        * **auto_adjust parameters**: they represent information related to the automatic learning rate control. 
        * **auto_adjust_lr**: the automatic learning rate adjustment algorithm to use. Valid values are None (default, don't auto adjust learning rate), AdjustAfterEpoch (check the training criterion after each epoch using the development set of the training set and decide whether to adjust the learning rate), and SearchBeforeEpoch (search the learning rate based on a small portion of the training set before each epoch starts).
    * When used in the AdjustAfterEpoch mode
        * **reduce_learn_rate_if_improve_less_than**: reduce the learning rate if the improvement is less than this value. Default is 0.
        * **learn_rate_decrease_factor**: the learning rate decrease factor. Default value is 0.618.
        * **increase_learn_rate_if_improve_more_than**: increase the learning rate if the improvement is larger than this value. Default value is `1#INF` (infinity) which means never increase.
        * **learn_rate_increase_factor**: the learning rate increase factor. Default value is 1.382.
        * **load_best_model**: weather to load the best model if the current model decreases the performance. Valid values are true (default) and false.
        * **learn_rate_adjust_interval**: determine the frequency of applying the learning rate adjustment check. Default is 1 epoch. If this value is set to a value larger than 1 the learning rate adjustment will be based on the average criterion computed from the last learnRateAdjustInterval epochs.
    
    * When used in the SearchBeforeEpoch mode.
        * **numMiniBatch4LRSearch**: the number of minibatches used to search the learning rate. Default value is 500. It's typically set to 10-20% of the total minibatches in an epoch.
        * **num_prev_learn_rate**: number of previous learning rates used as a hint to the search range. Default value is 5.
        * **num_best_search_epoch**: number of epochs in which we use the best learning rate instead of the sufficient learning rate . Default value is 1.
    
    * When used in the 'AdaptiveMinibatchSizing' mode
        * **num_minibatch_for_lr_search**: the number of minibatches used to search the minibatch size when in adaptive minibatch size mode. Default value is 500. It's typically set to 10-20% of the total minibatches in an epoch this is shared with the search for learning rate in SearchBeforeEpoch mode.
        * **auto_adjust_minibatch: enable or disable whether minibatch size is adaptively adjusted. Default value is false. Adapative minibatch sizing will begin on epochs starting after user minbatch sizes expcitily specified are complete. For example if the user specifed minibatchSize=256**:1024, then 256 and 1024 are used in the first 2 Epochs and adaptive minibatch sizing is used afterwards.
        * **minibatch_size_tuning_frequency**: The number of epochs to skip, on a periodic basis, before dynamically adjusting the minibatch size. Default value is 1.
        * **minibatch_size_tuning_max**: The maximum size allowed for an adaptively adjusted minibatch size. Default value is 1048576.

    * **continue_reduce**: If true, the learning rate is always reduced per epoch once it is reduced.
    * **num_prev_learn_rates**: TBA
    
    * Gradient control
        * **gradient_clipping_with_truncation**: whether to use the truncation based gradient clipping to control gradient explosion. Valid values are true (default) and false. If it is false the norm based clipping will be used instead which is more expensive.
        * **clipping_threshold_per_sample**: the clipping thread for each sample. Default value is `1#INF` which means infinity (i.e., clipping is turned off).
        * **L2_reg_weight**: the L2 regularization weight. Default is 0.
        * **L1_reg_weight**: the L1 regularization weight. Default is 0.
        * **grad_update_type**: gradient update type. Valid values are None (default, no special treatment to the gradient), AdaGrad, and RmsProp. 
      
        * When gradUpdateType equals to AdaGrad or RmsProp, you can control the behavior of the gradient update using following parameters:
            * **norm_with_ave_multiplier**: normalize the gradient with the average multipliers applied to the gradients by the AdaGrad/RmsProp algorithm. Default is true. 
      
        * When gradUpdateType equals to RmsProp, you can control the behavior of the gradient update using following parameters:
            * **rms_wgt_inc**: multiplicative increment of the learning rate scale. Default is 1.2.
            * **rms_wgt_dec**: multiplicative decrement of the learning rate scale. Default is 0.75.
            * **rms_wgt_max**: maximum learning rate scale allowed. A value closer to 1 makes the learning rate adjustment more stable but slower. Default is 10.
            * **rms_wgt_min**: minimum learning rate scale allowed. A value closer to 1 makes the learning rate adjustment more stable but slower. Default is 0.1.
            * **rms_gamma**: smoothing factor used to estimate the moving average of the variance. The smaller the value, the quicker it forgets the past information. Default is 0.99.
    
        * **gaussian_noise_inject_std**: the standard deviation of the Gaussian noise added when using the AdaGrad approach. Default is 0.
    
    * Adaptation
        * Only KL divergence regularization is directly supported. Other adaptation techniques can be easily implemented by adding computation nodes to the network using the model editing language (MEL). 
        * **adaptation_reg_type**: adaptation regularization type. Valid values are None (default) and KL (for KL divergence based regularization).
        * **adaptation_reg_weight**: adaptation regularization weight. Default value is 0.
    
    * Information display
        * **trace_level**: trace level to decide what information to print out in the stderr. Valid values are 0 (default) and 1.
        * **num_mbs_to_show_result**: display training statistics after how many minibatches. Default is 10.
        * **first_mbs_to_show_result**: the number of mini batches (counting from the start) to display training statistics for. 
        * **trace_node_names_real**: tracing (enable these for debugging) on the given nodes for real printing
        * **trace_node_names_category**: tracing (enable these for debugging) on the given nodes for category printing
        * **trace_node_names_sparse**: tracing (enable these for debugging) on the given nodes for sparse printing
    
    * Gradient Check
        * **gradient_check**: determines whether to use the gradient checker. The default value is false. When using the gradient checker you need to use a minibatch size that is larger than the sequence length for RNNs due to the truncated backpropagation through time (BPTT) algorithm used to train RNNs, and a smaller learning rate to prevent numerical issues caused by divergence. In addition, precision should be set to double.
    """
    def __init__(self,
                model_path=None,
                train_criterion_node_name=None,
                eval_criterion_node_name=None,
                epoch_size=None,
                keep_check_point_files=None,
                max_epochs=None,
                minibatch_size=None,
                dropout_rate=None,
                max_temp_mem_size_in_samples_for_cnn=None,
                learning_rates_per_mb=None,
                learning_rates_per_sample=None,
                momentum_per_mb=None,
                momentum_per_sample=None,
                momentum_as_time_constant=None,
                auto_adjust_lr=None,
                reduce_learn_rate_if_improve_less_than=None,
                learn_rate_decrease_factor=None,
                increase_learn_rate_if_improve_more_than=None,
                learn_rate_increase_factor=None,
                load_best_model=None,
                learn_rate_adjust_interval=None,
                num_prev_learn_rate=None,
                num_best_search_epoch=None,
                num_minibatch_for_lr_search=None,
                auto_adjust_minibatch=None,
                minibatch_size_tuning_frequency=None,
                minibatch_size_tuning_max=None,
                continue_reduce=None,
                num_prev_learn_rates=None,
                gradient_clipping_with_truncation=None,
                clipping_threshold_per_sample=None,
                L2_reg_weight=None,
                L1_reg_weight=None,
                grad_update_type=None,
                norm_with_ave_multiplier=None,
                rms_wgt_inc=None,
                rms_wgt_dec=None,
                rms_wgt_max=None,
                rms_wgt_min=None,
                rms_gamma=None,
                gaussian_noise_inject_std=None,
                adaptation_reg_type=None,
                adaptation_reg_weight=None,
                trace_level=None,
                num_mbs_to_show_result=None,
                first_mbs_to_show_result=None,
                trace_node_names_real=None,
                trace_node_names_category=None,
                trace_node_names_sparse=None,
                gradient_check=None,
                ):

        
        # this can be automated but cases like: maxTempMemSizeInSamplesForCNN
        # would be tricky because it is not camelCase
        self._py_to_cntk = {
        'model_path':'modelPath',
        'train_criterion_node_name':'trainCriterionNodeName',
        'eval_criterion_node_name':'evalCriterionNodeName',
        'epoch_size':'epochSize',
        'keep_check_point_files':'keepCheckPointFiles',
        'max_epochs':'maxEpochs',
        'minibatch_size':'minibatchSize',
        'dropout_rate':'dropoutRate',
        'max_temp_mem_size_in_samples_for_cnn':'maxTempMemSizeInSamplesForCNN',
        'learning_rates_per_mb':'learningRatesPerMB',
        'learning_rates_per_sample':'learningRatesPerSample',
        'momentum_per_mb':'momentumPerMB',
        'momentum_per_sample':'momentumPerSample',
        'momentum_as_time_constant':'momentumAsTimeConstant',
        'auto_adjust_lr':'autoAdjustLR',
        'reduce_learn_rate_if_improve_less_than':'reduceLearnRateIfImproveLessThan',
        'learn_rate_decrease_factor':'learnRateDecreaseFactor',
        'increase_learn_rate_if_improve_more_than':'increaseLearnRateIfImproveMoreThan',
        'learn_rate_increase_factor':'learnRateIncreaseFactor',
        'load_best_model':'loadBestModel',
        'learn_rate_adjust_interval':'learnRateAdjustInterval',
        'num_prev_learn_rate':'numPrevLearnRate',
        'num_best_search_epoch':'numBestSearchEpoch',
        'num_minibatch_for_lr_search':'numMiniBatch4LRSearch',
        'auto_adjust_minibatch':'autoAdjustMinibatch',
        'minibatch_size_tuning_frequency':'minibatchSizeTuningFrequency',
        'minibatch_size_tuning_max':'minibatchSizeTuningMax',
        'continue_reduce':'continueReduce',
        'num_prev_learn_rates':'numPrevLearnRates',
        'gradient_clipping_with_truncation':'gradientClippingWithTruncation',
        'clipping_threshold_per_sample':'clippingThresholdPerSample',
        'L2_reg_weight':'L2RegWeight',
        'L1_reg_weight':'L1RegWeight',
        'grad_update_type':'gradUpdateType',
        'norm_with_ave_multiplier':'normWithAveMultiplier',
        'rms_wgt_inc':'rms_wgt_inc',
        'rms_wgt_dec':'rms_wgt_dec',
        'rms_wgt_max':'rms_wgt_max',
        'rms_wgt_min':'rms_wgt_min',
        'rms_gamma':'rms_gamma',
        'gaussian_noise_inject_std':'gaussianNoiseInjectStd',
        'adaptation_reg_type':'adaptationRegType',
        'adaptation_reg_weight':'adaptationRegWeight',
        'trace_level':'traceLevel',
        'num_mbs_to_show_result':'numMBsToShowResult',
        'first_mbs_to_show_result':'firstMBsToShowResult',
        'trace_node_names_real':'traceNodeNamesReal',
        'trace_node_names_category':'traceNodeNamesCategory',
        'trace_node_names_sparse':'traceNodeNamesSparse',
        'gradient_check':'gradientCheck',
        }

        self._auto_adjust_params = [
                'auto_adjust_lr',
                'reduce_learn_rate_if_improve_less_than',
                'learn_rate_decrease_factor',
                'increase_learn_rate_if_improve_more_than',
                'learn_rate_increase_factor',
                'load_best_model',
                'learn_rate_adjust_interval',
                'num_prev_learn_rate',
                'num_best_search_epoch',
                'num_minibatch_for_lr_search',
                'auto_adjust_minibatch',
                'minibatch_size_tuning_frequency',
                'minibatch_size_tuning_max',
                'continue_reduce',
                'num_prev_learn_rates']
                
        self.model_path = model_path
        self.train_criterion_node_name = train_criterion_node_name
        self.eval_criterion_node_name = eval_criterion_node_name
        self.epoch_size = epoch_size
        self.keep_check_point_files = keep_check_point_files
        self.max_epochs = max_epochs
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        self.max_temp_mem_size_in_samples_for_cnn = max_temp_mem_size_in_samples_for_cnn
        self.learning_rates_per_mb = learning_rates_per_mb
        self.learning_rates_per_sample = learning_rates_per_sample
        self.momentum_per_mb = momentum_per_mb
        self.momentum_per_sample = momentum_per_sample
        self.momentum_as_time_constant = momentum_as_time_constant
        self.auto_adjust_lr = auto_adjust_lr
        self.reduce_learn_rate_if_improve_less_than = reduce_learn_rate_if_improve_less_than
        self.learn_rate_decrease_factor = learn_rate_decrease_factor
        self.increase_learn_rate_if_improve_more_than = increase_learn_rate_if_improve_more_than
        self.learn_rate_increase_factor = learn_rate_increase_factor
        self.load_best_model = load_best_model
        self.learn_rate_adjust_interval = learn_rate_adjust_interval
        self.num_prev_learn_rate = num_prev_learn_rate
        self.num_best_search_epoch = num_best_search_epoch
        self.num_minibatch_for_lr_search = num_minibatch_for_lr_search
        self.auto_adjust_minibatch = auto_adjust_minibatch
        self.minibatch_size_tuning_frequency = minibatch_size_tuning_frequency
        self.minibatch_size_tuning_max = minibatch_size_tuning_max
        self.continue_reduce = continue_reduce
        self.num_prev_learn_rates = num_prev_learn_rates
        self.gradient_clipping_with_truncation = gradient_clipping_with_truncation
        self.clipping_threshold_per_sample = clipping_threshold_per_sample
        self.L2_reg_weight = L2_reg_weight
        self.L1_reg_weight = L1_reg_weight
        self.grad_update_type = grad_update_type
        self.norm_with_ave_multiplier = norm_with_ave_multiplier
        self.rms_wgt_inc = rms_wgt_inc
        self.rms_wgt_dec = rms_wgt_dec
        self.rms_wgt_max = rms_wgt_max
        self.rms_wgt_min = rms_wgt_min
        self.rms_gamma = rms_gamma
        self.gaussian_noise_inject_std = gaussian_noise_inject_std
        self.adaptation_reg_type = adaptation_reg_type
        self.adaptation_reg_weight = adaptation_reg_weight
        self.trace_level = trace_level
        self.num_mbs_to_show_result = num_mbs_to_show_result
        self.first_mbs_to_show_result = first_mbs_to_show_result
        self.trace_node_names_real = trace_node_names_real
        self.trace_node_names_category = trace_node_names_category
        self.trace_node_names_sparse = trace_node_names_sparse
        self.gradient_check = gradient_check
        self.parallel_training = None
        
    def _set_global_parallel_params(self, 
                                    parallalization_method = None, 
                                    parallelization_start_epoch = None,
                                    distributed_mb_reading = None,
                                    sync_perf_stats = None):
        self.parallel_training = {
            'parallelizationMethod':parallalization_method,
            'parallelizationStartEpoch':parallelization_start_epoch,
            'distributedMBReading':distributed_mb_reading,
            'syncPerfStats':sync_perf_stats}
            
    def set_parallel_to_data_parallel(self, 
                                      parallelization_start_epoch = None,
                                      distributed_mb_reading = None,
                                      sync_perf_stats = None,
                                      gradient_bits = None,
                                      use_zero_threshold_for_1bit = None,
                                      use_buffered_async_gradient_aggregation = None):

        """
        This function sets the parallel training to Data Paralllel SGD.
                
        Args:
            parallelization_start_epoch (int): accepts integer value; default is 1
            distributed_mb_reading (bool): default is False It is recommended to 
                turn distributed minibatch reading on to minimize the I/O cost in each worker. 
            sync_perf_stats (int): accepts integer value; default is 0
            gradient_bits (int): the number of bits used to send gradient updates
            use_zero_threshold_for_1bit: TBA
            use_buffered_async_gradient_aggregation: TBA
        """
        self._set_global_parallel_params('DataParallelSGD',
                                         parallelization_start_epoch,
                                         distributed_mb_reading,
                                         sync_perf_stats)
        
        self.parallel_training_subblock = {
            'gradientBits':gradient_bits,        
            'useZeroThresholdFor1BitQuantization':use_zero_threshold_for_1bit,
            'useBufferedAsyncGradientAggregation':use_buffered_async_gradient_aggregation}
            
    def set_parallel_to_model_average(self, 
                                      parallelization_start_epoch = None,
                                      distributed_mb_reading = None,
                                      sync_perf_stats = None,
                                      sync_period = None,
                                      sync_frequency_in_frames = None):
        """
        This function sets the parallel training to Model Averaging SGD.
                
        Args:
            parallelization_start_epoch (int): accepts integer value; default is 1
            distributed_mb_reading (int): accepts boolean value:  True  or  False ; 
                default is False It is recommended to turn distributed minibatch 
                reading on to minimize the I/O cost in each worker. 
            sync_perf_stats (int): accepts integer value; default is 0
            sync_period (int): specifies the number of samples that each worker need 
                to process before a model averaging is conducted. The default value is 40,000.
            sync_frequency_in_frames: TBA       
        """        
        self._set_global_parallel_params('ModelAveragingSGD',
                                         parallelization_start_epoch,
                                         distributed_mb_reading,
                                         sync_perf_stats)
        
        self.parallel_training_subblock = {
            'syncPeriod':sync_period,                    
            'syncFrequencyInFrames':sync_frequency_in_frames}
                                              
    def set_parallel_to_block_momentum(self, 
                                      parallelization_start_epoch = None,
                                      distributed_mb_reading = None,
                                      sync_perf_stats = None,
                                      sync_period = None,
                                      reset_sgd_momentum = None,
                                      use_nesterov_momentum = None,
                                      block_learning_rate = None,
                                      block_momentum_per_sync = None,
                                      block_momentum_as_time_constant = None):
        """
        This function sets the parallel training to Block Momentum SGD.
                
        Args:
            parallelization_start_epoch (int): accepts integer value; default is 1
            distributed_mb_reading (bool): accepts boolean value:  True  or  False ; 
                default is False It is recommended to turn distributed minibatch 
                reading on to minimize the I/O cost in each worker. 
            sync_perf_stats (int): accepts integer value; default is 0
            sync_period: it specifies how frequent a model synchronization is performed. 
                The default value is 120,000.
            reset_sgd_momentum (bool): This means after every synchronization point, 
                the smoothed gradient used in local SGD will be set as 0. The default
                value of this variable is True. 
            use_nesterov_momentum (bool): This means the Nestrov style block momentum 
                is applied. The default value of this variable is True. 
            block_learning_rate (float): specifies the block learning rate. 
            block_momentum_per_sync: TBA
            block_momentum_as_time_constant (float): specifies the time constant of the 
                low-pass filter in block-level model update. It is calculated as: 
                blockMomentumAsTimeConstant = -syncPeriod / log(block_momentum). 
                Note that block_momentum_per_sync and block_momentum_as_time_constant 
                are mutually exclusive
        
        """        
        self._set_global_parallel_params('BlockMomentumSGD',
                                         parallelization_start_epoch,
                                         distributed_mb_reading,
                                         sync_perf_stats)
        
        self.parallel_training_subblock = {
            'syncPeriod':sync_period,        
            'resetSGDMomentum':reset_sgd_momentum,
            'useNesterovMomentum':use_nesterov_momentum,
            'blockLearningRate':block_learning_rate,
            'blockMomentumPerSync':block_momentum_per_sync,
            'blockMomentumAsTimeConstant':block_momentum_as_time_constant}


    def _generate_parallel_training_config(self):
        config = ['ParallelTrain=[']        
        for k,v in self.parallel_training.items():
            if v is not None:
                config.append('\t{0} = {1}'.format(k, v))    
        
        config.append('\t{0} = ['.format(self.parallel_training['parallelizationMethod']))    
        for k,v in self.parallel_training_subblock.items():            
            if v is not None:
                config.append('\t\t{0} = {1}'.format(k, v))    
        config.append('\t]')
        config.append(']')
        return '\n'.join(config)
        
    def _to_config_description(self):
        """Generate the SGDParams configuration block
        """
        config = []
        auto_adjust_block = []
        for k, v in self.__dict__.items():
            if  not k.startswith('parallel_training') and k[0] != '_' and v is not None:
                # this is a sub-block. 
                #TODO: perhaps move this to a separete method (set_auto_adjust),
                # but then the user would need to call it explicitly 
                if k in self._auto_adjust_params:
                    auto_adjust_block.append('\t{0} = {1}\n'.format(self._py_to_cntk[k], v))
                else:
                    config.append('{0} = {1}\n'.format(self._py_to_cntk[k], v))    
            
        if len(auto_adjust_block) > 0:
            config.append("autoAdjust=[\n")
            config.extend(auto_adjust_block)
            config.append("\t]")
            
        if self.parallel_training:
            config.append(self._generate_parallel_training_config())
            
        return ''.join(config)
