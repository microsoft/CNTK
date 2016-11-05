import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging

from models import *
from ferplus import *

from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.ops import cross_entropy_with_softmax, classification_error
from cntk.ops import input_variable, constant, parameter, softmax
import cntk as ct

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid'] 
test_folders  = ['FER2013Test']

def cost_func(training_mode, prediction, target):
    '''
    We use cross entropy in most mode, except for the multi-label mode, which require treating
    multiple labels exactly the same.
    '''
    train_loss = None

    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross Entropy.
        train_loss = ct.reduce_sum(ct.minus(ct.reduce_log_sum(prediction, axis=0), ct.reduce_sum(ct.element_times(target, prediction), axis=0)))
    elif training_mode == 'multi_target':
        train_loss = ct.negate(ct.log(ct.reduce_max(ct.element_times(target, prediction), axis=0)))

    return train_loss
    
def main(base_folder, training_mode='majority', model_name='vgg13', max_epochs = 100):
    # create needed folders.
    output_model_path   = os.path.join(base_folder, R'models')
    output_model_folder = os.path.join(output_model_path, model_name)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # creating logging file 
    logging.basicConfig(filename = os.path.join(output_model_folder, "train.log"), filemode = 'w', level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    # create the model
    num_classes = len(emotion_table)
    model       = build_model(num_classes, model_name)

    # set the input variables.
    input_var = input_variable((1, model.input_height, model.input_width), np.float32)
    label_var = input_variable((num_classes), np.float32)
    
    # read FER+ dataset.
    logging.info("Loading data...")
    train_params        = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True)

    train_data_reader   = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader     = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader    = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)
    
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    # get the probalistic output of the model.
    z    = model.model(input_var)
    pred = softmax(z)
    
    epoch_size = train_data_reader.size()
    minibatch_size = 32

    # Training config
    lr_schedule            = [model.learning_rate]*20 + [model.learning_rate / 2.0]*20 + [model.learning_rate / 10.0]
    lr_per_minibatch       = learning_rate_schedule(lr_schedule, epoch_size, UnitType.minibatch)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(model.momentum))
    
    # loss and error cost
    train_loss = cost_func(training_mode, pred, label_var)
    pe         = classification_error(pred, label_var)

    # construct the trainer
    learner = momentum_sgd(pred.parameters, 
                           lr = lr_per_minibatch, momentum = momentum_time_constant)
    trainer = Trainer(z, train_loss, pe, learner)

    # Get minibatches of images to train with and perform model training
    max_val_accuracy    = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy  = 0.0

    logging.info("Start training...")
    epoch = 0
    while epoch < max_epochs: 
        train_data_reader.reset()
        val_data_reader.reset()
        test_data_reader.reset()
        
        # Training 
        start_time = time.time()
        training_loss = 0
        training_accuracy = 0
        while train_data_reader.has_more():
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({input_var : images, label_var : labels})

            # keep track of statistics.
            training_loss     += get_train_loss(trainer) * current_batch_size
            training_accuracy += get_train_eval_criterion(trainer) * current_batch_size
                
        training_accuracy /= train_data_reader.size()
        training_accuracy = 1.0 - training_accuracy
        
        # Validation
        val_accuracy = 0
        while val_data_reader.has_more():
            images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
            val_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
        val_accuracy /= val_data_reader.size()
        val_accuracy = 1.0 - val_accuracy
        
        # if validation accuracy goes higher, we compute test accuracy
        test_run = False
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            test_run = True
            test_accuracy = 0
            while test_data_reader.has_more():
                images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                test_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
            test_accuracy /= test_data_reader.size()
            test_accuracy = 1.0 - test_accuracy
            final_test_accuracy = test_accuracy
            if final_test_accuracy > best_test_accuracy: 
                best_test_accuracy = final_test_accuracy
 
        logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
        logging.info("  training loss:\t{:e}".format(training_loss))
        logging.info("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
        logging.info("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
        if test_run:
            logging.info("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
            
        epoch += 1

    logging.info("")
    logging.info("Best validation accuracy:\t\t{:.2f} %".format(max_val_accuracy * 100))
    logging.info("Test accuracy corresponding to best validation:\t\t{:.2f} %".format(final_test_accuracy * 100))
    logging.info("Best test accuracy:\t\t{:.2f} %".format(best_test_accuracy * 100))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type = str, 
                        help = "Base folder containing the training, validation and testing data.", 
                        required = True)
    parser.add_argument("-m", 
                        "--training_mode", 
                        type = str,
                        default='crossentropy', 
                        help = "Specify the training mode: majority, probability, crossentropy or multi_target.")

    args = parser.parse_args()
    main(args.base_folder, args.training_mode)