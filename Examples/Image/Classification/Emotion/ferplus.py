import sys
import os
import csv
import numpy as np
import logging
import random as rnd
from collections import namedtuple

from PIL import Image
import img_util as imgu
import matplotlib.pyplot as plt

def display_summary(train_data_reader, val_data_reader, test_data_reader):
    '''
    Summarize the data in a tabular format.
    '''
    emotion_count = train_data_reader.emotion_count
    emotin_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

    logging.info("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))
    for index in range(emotion_count):
        logging.info("{0}\t{1}\t{2}\t{3}".format(emotin_header[index].ljust(10), 
                     train_data_reader.per_emotion_count[index], 
                     val_data_reader.per_emotion_count[index], 
                     test_data_reader.per_emotion_count[index]))

class FERPlusParameters():
    '''
    FER+ reader parameters
    '''
    def __init__(self, target_size, width, height, training_mode = "majority", determinisitc = False, shuffle = True):
        self.target_size   = target_size
        self.width         = width
        self.height        = height
        self.training_mode = training_mode
        self.determinisitc = determinisitc
        self.shuffle       = shuffle
                     
class FERPlusReader(object):
    '''
    A custom reader for FER+ dataset that support multiple modes as discribed in:
        https://arxiv.org/abs/1608.01041
    '''
    @classmethod
    def create(cls, base_folder, sub_folders, label_file_name, parameters):
        '''
        Factory function that create an instance of EmotionDataReader and load the data form disk.
        '''
        reader = cls(base_folder, sub_folders, label_file_name, parameters)
        reader.load_folders(parameters.training_mode)
        return reader
        
    def __init__(self, base_folder, sub_folders, label_file_name, parameters):
        self.base_folder     = base_folder
        self.sub_folders     = sub_folders
        self.label_file_name = label_file_name
        self.emotion_count   = parameters.target_size
        self.width           = parameters.width
        self.height          = parameters.height
        self.shuffle         = parameters.shuffle
        self.training_mode   = parameters.training_mode

        # data augmentation parameters.determinisitc
        if parameters.determinisitc:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew = 0.0
            self.do_flip = False
        else:
            self.max_shift = 0.08
            self.max_scale = 1.05
            self.max_angle = 20.0
            self.max_skew = 0.05
            self.do_flip = True
        
        self.data              = None
        self.per_emotion_count = None
        self.batch_start       = 0
        self.indices           = 0

        self.A, self.A_pinv = imgu.compute_norm_mat(self.width, self.height)
        
    def has_more(self):
        if self.batch_start < len(self.data):
            return True
        return False

    def reset(self):
        self.batch_start = 0

    def size(self):
        return len(self.data)
        
    def next_minibatch(self, batch_size):
        data_size = len(self.data)
        batch_end = min(self.batch_start + batch_size, data_size)
        current_batch_size = batch_end - self.batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')
        
        inputs = np.empty(shape=(current_batch_size, 1, self.width, self.height), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.emotion_count), dtype=np.float32)
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            distorted_image = imgu.distort_img(self.data[index][1], self.width, self.height, self.max_shift, self.max_scale, self.max_angle, self.max_skew, self.do_flip)
            final_image = imgu.preproc_img(distorted_image, A=self.A, A_pinv=self.A_pinv)

            inputs[idx-self.batch_start]    = final_image
            targets[idx-self.batch_start,:] = self.__process_target(self.data[index][2])

        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size
        
    def load_folders(self, mode):
        self.reset()
        self.data = []
        self.per_emotion_count = np.zeros(self.emotion_count, dtype=np.int)
        
        for folder_name in self.sub_folders: 
            logging.info("Loading %s" % (os.path.join(self.base_folder, folder_name)))
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile: 
                emotion_label = csv.reader(csvfile) 
                for row in emotion_label: 
                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    image_data = Image.open(image_path)
                    image_data.load()
                    
                    emotion_raw = list(map(float, row[2:len(row)]))
                    emotion = self.__process_data(emotion_raw, mode) 
                    idx = np.argmax(emotion)
                    if idx < self.emotion_count: # not unknown or non-face 
                        emotion = emotion[:-2]
                        emotion = [float(i)/sum(emotion) for i in emotion]
                        self.data.append((image_path, image_data, emotion))
                        self.per_emotion_count[idx] += 1
        
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __process_target(self, target):
        if self.training_mode == 'majority' or self.training_mode == 'crossentropy': 
            return target
        elif self.training_mode == 'probability': 
            idx             = np.random.choice(len(target), p=target) 
            new_target      = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target': 
            new_target = np.array(target) 
            new_target[new_target>0] = 1.0
            epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)

    def __process_data(self, emotion_raw, mode): 
        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal) 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            # find the peak value of the emo_raw list 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   # force setting as unknown 
        elif (mode == 'probability') or (mode == 'crossentropy'):         
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw) 
                for i in range(size): 
                    if emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes 
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown 
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
                emotion = emotion_unknown   # set as unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]