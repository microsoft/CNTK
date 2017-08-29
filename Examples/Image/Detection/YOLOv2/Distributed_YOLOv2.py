# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import argparse
import _cntk_py
import cntk

from cntk.io import FULL_DATA_SWEEP
from cntk import *
from cntk.logging import ProgressPrinter
from cntk.train.distributed import Communicator
from numpy.distutils.lib2def import output_def

import YOLOv2 as yolo2
from TrainUDFyolov2 import *
from PARAMETERS import *
import PARAMETERS as par

import pdb
from cntk_debug import DebugLayer
from cntk_debug_single import DebugLayerSingle

model_path = None
# Create a minibatch source.
def create_image_mb_source(image_file, gtb_file, is_training, total_number_of_samples):

    return yolo2.create_mb_source(par_image_height, par_image_width, par_num_channels, (5 * par_max_gtbs), image_file,
                                        gtb_file,
                                        multithreaded_deserializer=True,
                                        is_training=is_training,
                                        max_samples=total_number_of_samples)


# Create trainer
def create_trainer(to_train, epoch_size, minibatch_size, num_quantization_bits, printer, block_size, warm_up):
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    lr_schedule = cntk.learning_rate_schedule(par_lr_schedule, cntk.learners.UnitType.sample,
                                              epoch_size)


    mm_schedule = cntk.learners.momentum_as_time_constant_schedule([-minibatch_size / np.log(par_momentum)])

    # Instantiate the trainer object to drive the model training
    local_learner = cntk.learners.momentum_sgd(to_train['output'].parameters, lr_schedule, mm_schedule, unit_gain=False,
                                         l2_regularization_weight=par_weight_decay)

    # Create trainer
    if block_size != None:
        parameter_learner = block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = data_parallel_distributed_learner(local_learner,
                                                              num_quantization_bits=num_quantization_bits,
                                                              distributed_after=warm_up)

    return cntk.Trainer(None, (to_train['mse'], to_train['mse']), parameter_learner, printer)


# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size,
        epoch_size, model_path, restore):

    input_map = {
        network['feature']: train_source["features"],
        network['gtb_in']: train_source["label"]
    }


    callback_frequency = par_save_after_each_n_epochs

    #'index', 'average_error', 'cv_num_samples', and 'cv_num_minibatches'
    def safe_model_callback(index, average_error, cv_num_samples, cv_num_minibatches):
        if(Communicator.rank()!=0 or model_path is None or callback_frequency is None):return True
        callback_save_file = os.path.join(model_path,
                                          "after_" + str(callback_frequency * (index + 1)) + "_epochs.model")
        print(Communicator.rank())
        network['output'].save(callback_save_file)
        print("Saved intermediate model to " + callback_save_file)
        return True

    model_name = "checkpoint"


    #test_config = TestConfig(source=test_source, mb_size=minibatch_size) if test_source else None
    #checkpoint_config = CheckpointConfig(filename=os.path.join(model_path, model_name),
    #                                       frequency=5000,
    #                                       preserve_all=True,
    #                                       restore=restore) if model_path else None
    cv_config = cntk.CrossValidationConfig(None, mb_size=par_minibatch_size, frequency=1,
                                           callback=safe_model_callback)

    use_training_session = True
    if not use_training_session:
        for i in range(3):
            print("--- iteration {} ---".format(i))
            data = train_source.next_minibatch(2, input_map=input_map)  # fetch minibatch.
            trainer.train_minibatch({image_input: data[image_input].asarray(),
                                     gtb_input: (data[gtb_input]).asarray()}, device=gpu(0))

    # Train all minibatches
    else:
        training_session(
        trainer=trainer, mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size,
        progress_frequency=epoch_size,
        checkpoint_config= CheckpointConfig(filename=os.path.join(model_path, "Checkpoint_YOLOv2"), restore=False) if model_path is not None else None,
        test_config=TestConfig(source=test_source, mb_size=minibatch_size) if test_source else None,
        cv_config=cv_config
    ).train()

    #WH_out = network['mse'].find_by_name('WH-Out')
    #feat = cntk.combine([WH_out]).eval(test_features)
    #print("raw cntk: {}".format(feat[0, 527, :]))


# Train and evaluate the network.
def yolov2_train_and_eval(network,
                          train_image_file, train_gtb_file,
                          test_image_file, test_gtb_file,
                          num_quantization_bits=32, block_size=3200, warm_up=0,
                          minibatch_size=64, epoch_size=5000, max_epochs=1,
                          restore=True, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=True):
    _cntk_py.set_computation_network_trace_level(0)

    progress_printer = ProgressPrinter(
        freq=num_mbs_per_log if num_mbs_per_log is not None else int(epoch_size/10),
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        num_epochs=max_epochs,
        test_freq=1)


    trainer = create_trainer(network, epoch_size, minibatch_size, num_quantization_bits, progress_printer, block_size, warm_up)
    train_source = create_image_mb_source(train_image_file, train_gtb_file, True, total_number_of_samples=max_epochs * epoch_size)
    if test_image_file or test_gtb_file:
        test_source = create_image_mb_source(test_image_file, test_gtb_file, False, total_number_of_samples=FULL_DATA_SWEEP)
    else:
        test_source = None
    train_and_test(network, trainer, train_source, test_source, minibatch_size,
            epoch_size, model_path, restore)

    return network['output']


if __name__ == '__main__':
    DETERMINISTIC = False

    if DETERMINISTIC:
        from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms

        set_fixed_random_seed(1)
        force_deterministic_algorithms()


    parser = argparse.ArgumentParser()

    if par.par_dataset_name == "Pascal_VOC_2007":data_path = os.path.join(par_abs_path, "..", "..", "DataSets","Pascal", "mappings")
    elif par.par_dataset_name == "Grocery":data_path = os.path.join(par_abs_path, "..", "..", "DataSets", "Grocery")
    elif par.par_dataset_name == "Overfit":data_path = os.path.join(par_abs_path, "..", "..", "DataSets", "Overfit")
    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located',required=False, default=data_path)

    parser.add_argument('-trainimages', '--trainimages', help='File containing the images in ImageReader format',
                        required=False, default=None)
    parser.add_argument('-traingts', '--traingts', help='File containing the bounding boxes and labels in CTF format',
                        required=False, default=None)
    parser.add_argument('-testimages', '--testimages', help='File containing the images in ImageReader format',
                        required=False, default=None)
    parser.add_argument('-testgts', '--testgts', help='File containing the bounding boxes and labels in CTF format',
                        required=False, default=None)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False,
                        default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False,
                        default=par_max_epochs)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default=par_minibatch_size)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default=par_epoch_size)
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation',
                        type=int, required=False, default='32')
    parser.add_argument('-r', '--restart',
                        help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)',
                        action='store_true')
    parser.add_argument('-d', '--devices', type=str, help="Specifies the devices for the individual workers. Negative numbers: CPU. Positive numbers: GPU Id.",
                        required=False, default=None)
    parser.add_argument('-b', '--block_samples', type=int,
                        help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)",
                        required=False, default=None)
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed',
                        type=int, required=False, default='0')

    args = vars(parser.parse_args())

    output_dir = os.path.join(".","outputdir")#None
        model_path = None
    if args['outputdir'] is not None:
        output_dir = args['outputdir']
        model_path = args['outputdir'] + "/models"
    else:
        model_path = None

    log_dir = args['logdir']
    if args['devices'] is not None:
        # Setting one worker on GPU and one worker on CPU. Otherwise memory consumption is too high for a single GPU.
        dev = [int(d) for d in args['devices'].split(',')][Communicator.rank()]
        if dev >= 0:
            dev = cntk.device.gpu(dev)
        else:
            dev = cntk.device.cpu()

        cntk.device.try_set_default_device(dev)

    data_path = args['datadir']

    train_image_file = args['trainimages']
    train_gt_file = args['traingts']
    if train_image_file is None or train_gt_file is None:
        if not os.path.isdir(data_path):
            raise RuntimeError("Directory %s does not exist" % data_path)
        train_image_file = os.path.join(data_path, par_train_data_file)
        train_gt_file = os.path.join(data_path, par_train_roi_file)

    test_image_file = args['testimages']
    test_gt_file = args['testgts']

    ####################################################################################################################
    model = yolo2.create_yolov2_net(par)

    image_input = input_variable((par_num_channels, par_image_height, par_image_width), name="data")

    output = model(image_input)  # append model to image input

    # input for ground truth boxes
    num_gtb = par_max_gtbs
    gtb_input = input_variable((num_gtb * 5))  # 5 for class, x,y,w,h

    if not par_boxes_centered:
        original_shape = gtb_input.shape
        new_shape = (num_gtb, 5)
        reshaped = reshape(gtb_input, new_shape)
        xy = reshaped[:,0:2]
        wh = reshaped[:,2:4]
        cls = reshaped[:,4:]
        center_xy = xy + wh*.5
        new_gtb = splice(xy,wh,cls,axis=1)
        gtb_transformed = reshape(new_gtb, gtb_input.shape)
    else:
        gtb_transformed = gtb_input

    from ErrorFunction import get_error

    if False:
        output = user_function(DebugLayer(output, image_input, gtb_transformed, debug_name="out-img-gt"))
    mse = get_error(output, gtb_transformed, cntk_only=False)# + zero

    network = {
        'feature': image_input,
        'gtb_in': gtb_input,
        'mse': mse,
        'output': output
        #'trainfunction': ud_tf
    }

    ####################################################################################################################

    try:
        nr_of_epoch = args['num_epochs']

        output = yolov2_train_and_eval(network, 
                                       train_image_file, train_gt_file,
                                       test_image_file, test_gt_file,
                                       max_epochs=nr_of_epoch,
                                       restore=not args['restart'],
                                       log_to_file=log_dir,
                                       num_mbs_per_log=50,
                                       num_quantization_bits=args['quantized_bits'],
                                       block_size=args['block_samples'],
                                       warm_up=args['distributed_after'],
                                       minibatch_size=args['minibatch_size'],
                                       epoch_size=args['epoch_size'],
                                       gen_heartbeat=True)

    finally:
        Communicator.finalize()
        print("Training finished!")

    if output is not None and output_dir is not None:
        save_path = os.path.join(output_dir, "YOLOv2.model")
        output.save(save_path)
        print("Saved model to " + save_path)
