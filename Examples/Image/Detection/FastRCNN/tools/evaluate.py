# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ============================================================================

from py_cpu_nms import *
import cv2, os
import matplotlib.pyplot as mp

from cntk import *

class ClassMap():
    """Awaits format {<Identifier> <number> }"""
    def __init__(self, cls_map_file):
        strings = open(cls_map_file).read().split()
        self.cls_map = {}
        for i in range(int(len(strings)/2)):
            self.cls_map[int(strings[2*i+1])] = strings[2*i]

    def getClass(self,i):
        return self.cls_map[i]

LIMIT_TO_FIRST = None
NMS_IOU_THRESHOLD = 0.7
cls_map = ClassMap(r"..\..\DataSets\Pascal\class_map.txt")
DATA_SET = "Overfit"
CONF_THRESHOLD = 0.15
cls_map = ClassMap(r"..\..\DataSets\Pascal\class_map.txt") if DATA_SET == "Pascal_VOC_2007" or DATA_SET=="Overfit"\
    else ClassMap(r"..\..\DataSets\Grocery\Class_map.txt")

def draw_bb_on_image(image, bb_list):
    image_width = len(image[1])
    image_height = len(image)

    box_list_len = min(len(bb_list), LIMIT_TO_FIRST) if LIMIT_TO_FIRST is not None else len(bb_list)
    for j in range(box_list_len):
        box = bb_list[j]
        xmin = int(image_width * (box[0] -  box[2] / 2))
        xmax = int(image_width * (box[0] +  box[2] / 2))
        ymin = int(image_height * (box[1] - box[3] / 2))
        ymax = int(image_height * (box[1] + box[3] / 2))
        if(xmax >= image_width or ymax >= image_height or xmin < 0 or ymin < 0):
            print("Box out of bounds: (" + str(xmin) +","+ str(ymin) +") ("+ str(xmax) +","+ str(ymax) +")")
            # print(box[5:])
        xmax = image_width-1 if xmax >= image_width else xmax
        ymax = image_height-1 if ymax >= image_height else ymax
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin

        color = (255, 255 - int(j*255/box_list_len), int(j*255/box_list_len))
        print((xmin, xmax, ymin, ymax, image_width, image_height))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        # print(box[5:])
        detected_class = np.argmax(box[5:]) + 1

        cv2.putText(image, cls_map.getClass(detected_class), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color,1)

    return image

def plot_image(image):
    mp.imshow(image)
    mp.plot()
    mp.show()

def xywh_to_point(xywh):
    points = np.zeros(xywh.shape)

    points[:, 0] = xywh[:, 0] - .5 * xywh[:, 2]  # x1
    points[:, 1] = xywh[:, 1] - .5 * xywh[:, 3]  # y1
    points[:, 2] = xywh[:, 0] + .5 * xywh[:, 2]  # x2
    points[:, 3] = xywh[:, 1] + .5 * xywh[:, 3]  # y2
    points[:, 4:] = xywh[:,4:]

    return points

def points_to_xywh(points):
    xywh = np.zeros(points.shape)

    xywh[:,0] = (points[:,0] + points[:,2])/2
    xywh[:, 1] = (points[:, 1] + points[:, 3]) / 2
    xywh[:,2] = np.abs(points[:,2] - points[:,0])
    xywh[:, 3] = np.abs(points[:, 3] - points[:, 1])
    xywh[:, 4:] = points[:,4:]

    return xywh

def do_nms(predictions):
    if NMS_IOU_THRESHOLD is not None:
        to_run = predictions[np.where(predictions[:,4]>CONF_THRESHOLD)]

        indicies = py_cpu_nms(xywh_to_point(to_run), NMS_IOU_THRESHOLD)
        return to_run[indicies]

    return predictions

def predictions_for_image(cv_img, model, input_width, input_height):
    resized = cv2.resize(cv_img, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    bgr_image = np.asarray(resized, dtype=np.float32)
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

    arguments = {model.arguments[0]: [hwc_format]}

    predictions = model.eval(arguments)[0]

    return predictions

def load_image(img_path):
    return cv2.imread(img_path)

def save_image(img, dir, name):
    cv2.imwrite(os.path.join(dir,name), img)

if __name__ == "__main__":

    model = load_model(os.path.join(".", "outputdir", r"YOLOv2.model"))
    data_input = logging.graph.find_by_name(model, "data")
    img_width = data_input.shape[2]
    img_height= data_input.shape[1]

    if DATA_SET == "Pascal_VOC_2007":
        obj_min, obj_max=1, 0
        data_path= r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages"
        img_list = [18,118,1118,27,2118,4118,1,2,3,4,5,6,7,8,9,10]
        # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
        save_path = os.path.join(".", "outputdir", "results", "pvoc2007")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(len(img_list)):
            img_name =  "{:06}.jpg".format(img_list[i])
            img = load_image(os.path.join(data_path, img_name))

            preds = predictions_for_image(img, model, img_width, img_height)
            preds_nms = do_nms(preds)
            #import ipdb;ipdb.set_trace()
            color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            draw_bb_on_image(color_image, preds_nms)

            if i<0:
                plot_image(color_image)

            out_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            save_image(out_img, save_path, "bb_"+img_name)

            import math
            objectnesses = preds[:,4]
            obj_min = np.minimum(obj_min, np.minimum.reduce(objectnesses))
            obj_max = np.maximum(obj_max, np.maximum.reduce(objectnesses))
        print((obj_min,obj_max))


    elif DATA_SET == "Grocery":
        data_path = r"..\..\DataSets\Grocery"
        img_list = ["positive\WIN_20160803_11_29_07_Pro",
                    "positive\WIN_20160803_11_30_07_Pro" ,
                    "testImages\WIN_20160803_11_28_42_Pro",
                    "testImages\WIN_20160803_11_42_36_Pro",
                    "testImages\WIN_20160803_11_46_03_Pro",
                    "testImages\WIN_20160803_11_48_26_Pro",
                    "testImages\WIN_20160803_12_37_07_Pro"]
        # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
        save_path = os.path.join(".", "outputdir", "results", "grocery")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        bla = 0
        for i in range(len(img_list)):
            img_name = img_list[i] + ".jpg"
            img = load_image(os.path.join(data_path, img_name))

            preds = predictions_for_image(img, model, img_width, img_height)
            preds_nms = do_nms(preds)
            # import ipdb;ipdb.set_trace()
            color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            draw_bb_on_image(color_image, preds_nms)

            if i < 0:
                plot_image(color_image)

            out_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            bla+=1
            save_image(out_img, save_path, "bb_" + str(bla)+".jpg")

    elif DATA_SET == "Overfit":
        obj_min, obj_max = 1, 0
        data_path = r"..\..\DataSets\Overfit"
        img_list = [37,38]
        # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
        save_path = os.path.join(".", "outputdir", "results", "overfit")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i in range(len(img_list)):
            img_name = "{:06}.jpg".format(img_list[i])
            img = load_image(os.path.join(data_path, img_name))

            preds = predictions_for_image(img, model, img_width, img_height)
            preds_nms = do_nms(preds)
            # import ipdb;ipdb.set_trace()
            color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            draw_bb_on_image(color_image, preds_nms)

            if i < 0:
                plot_image(color_image)

            out_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            save_image(out_img, save_path, "bb_" + img_name)

            import math

            objectnesses = preds[:, 4]
            obj_min = np.minimum(obj_min, np.minimum.reduce(objectnesses))
            obj_max = np.maximum(obj_max, np.maximum.reduce(objectnesses))
        print((obj_min, obj_max))