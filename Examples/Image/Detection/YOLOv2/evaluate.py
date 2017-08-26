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
        self.cls_map = [None]*int(len(strings)/2)#{}
        for i in range(int(len(strings)/2)):
            self.cls_map[int(strings[2*i+1])] = strings[2*i]

    def getClass(self,i):
        return self.cls_map[i]

LIMIT_TO_FIRST = 10
NMS_IOU_THRESHOLD = .3 # 0.7
cls_map = ClassMap(r"../../DataSets/Pascal/mappings/class_map.txt")
DATA_SET = "Pascal_VOC_2007"
CONF_THRESHOLD = 0.5 # 0.015
cls_map = ClassMap(r"../../DataSets/Pascal/mappings/class_map.txt") if DATA_SET == "Pascal_VOC_2007" or DATA_SET=="Overfit"\
    else ClassMap(r"../../DataSets/Grocery/Class_map.txt")

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
        #print((xmin, xmax, ymin, ymax, image_width, image_height))
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


def prepare_ground_truth_boxes(gtbs, classes, image_width, image_height):
    """
        Creates an object that can be passed as the parameter "all_gt_infos" to "evaluate_detections" in map_helpers
        Parameters
        ----------
        gtbs - arraylike of shape (nr_of_images, nr_of_boxes, cords+original_label) where nr_of_boxes may be a dynamic axis

        Returns
        -------
        Object for parameter "all_gt_infos"
        """
    num_test_images = len(gtbs)
    all_gt_infos = {key: [] for key in classes}
    for image_i in range(num_test_images):
        image_gtbs = np.copy(gtbs[image_i])
        coords = image_gtbs[:, 0:4]
        original_labels = image_gtbs[:, -1:]

        coords = xywh_to_point(coords)
        coords[:,[0,2]] *= image_width
        coords[:,[1,3]] *= image_height

        all_gt_boxes = np.concatenate([coords, original_labels], axis=1)

        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:, -1] == cls_index)]
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})

    return all_gt_infos

def prepare_predictions(outputs, classes,  image_width, image_height):
    """
        prepares the prediction for the ap computation.
        :param outputs: list of outputs per Image of the network
        :param roiss: list of rois rewponsible for the predictions of above outputs.
        :param num_classes: the total number of classes
        :return: Prepared object for ap computation by utils.map.map_helpers
        """
    num_test_images = len(outputs)

    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(len(classes))]

    for img_i in range(num_test_images):
        output = outputs[img_i]
        #output = output[0]
        coords = output[:,:4]
        objs = output [:,4:5]
        cls_preds = output[:,5:]
        labels = np.argmax(cls_preds, axis=1) + 1
        labels.shape += (1,)

        coords = xywh_to_point(coords)
        coords[:,[0, 2]] *= image_width
        coords[:,[1, 3]] *= image_height


        print(coords.shape, objs.shape, labels.shape)
        preds_for_img = np.concatenate([coords, objs, labels], axis=1)  # (nr_of_rois x 6) --> coords_score_label

        for cls_j in range(1, len(classes)):
            coords_score_label_for_cls = preds_for_img[np.where(preds_for_img[:, -1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:, :-1].astype(np.float32, copy=False)

    return all_boxes

def eval_map(model, img_file, gtb_file, num_images_to_eval):
    from YOLOv2 import create_mb_source
    abs_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(abs_path, ".."))
    from utils.map.map_helpers import evaluate_detections

    import PARAMETERS as par
    rois_per_image = par.par_max_gtbs

    data_input = logging.graph.find_by_name(model, "data")
    img_width = data_input.shape[2]
    img_height = data_input.shape[1]

    mb_source = create_mb_source(img_height=img_height, img_width=img_width, img_channels=3, output_size=rois_per_image*5, image_file=img_file, roi_file=gtb_file, is_training=False, max_samples=num_images_to_eval)

    image_input = input_variable((3, img_height, img_width),
                                 dynamic_axes=[Axis.default_batch_axis()])
    gt_input = input_variable((rois_per_image*5,))
    input_map = {  # add real gtb
        image_input: mb_source.streams.features,
        gt_input: mb_source.streams.label,
    }
    model = model(image_input)

    all_raw_gt_boxes = []
    all_raw_outputs = []
    all_raw_imgs = []

    VISUALIZE = False

    classes = cls_map.cls_map[1:]
    # evaluate test images and write network output to file
    print("Evaluating YOLOv2 model for %s images." % num_images_to_eval)
    #print(type(classes))
    for img_i in range(0, num_images_to_eval):
        mb_data = mb_source.next_minibatch(1, input_map=input_map)

        # receives rel coords
        gt_data = mb_data[gt_input].asarray()
        gt_data.shape = (rois_per_image, 5)


        all_gt_boxes = gt_data[np.where(gt_data[:, 4] != 0)]  # remove padded boxes!
        all_raw_gt_boxes.append(all_gt_boxes.copy())

        img = mb_data[image_input].asarray()

        if VISUALIZE:
            all_raw_imgs.append(img)

        output = model.eval({image_input: mb_data[image_input]})
        #import ipdb;ipdb.set_trace()
        output = output[0]
        all_raw_outputs.append(output[np.where(output[:, 4] > CONF_THRESHOLD )].copy())

        if img_i % 1000 == 0 and img_i != 0:
            print("Images processed: " + str(img_i))

    all_gt_infos = prepare_ground_truth_boxes(all_raw_gt_boxes, classes, img_width, img_height)
    all_boxes = prepare_predictions(all_raw_outputs, classes, img_width, img_height)

    if VISUALIZE:
        bb_img_gt_l = visualize_gt(all_gt_infos, all_raw_imgs, False)
        bb_img_rois_l = visualize_rois(all_boxes, all_raw_imgs, False)

        for img_i in range(len(bb_img_gt_l)):
            save_image(bb_img_gt_l[img_i], ".", "test_gt_" + str(img_i) + ".png")
        for img_i in range(len(bb_img_rois_l)):
            save_image(bb_img_rois_l[img_i], ".", "test_rois_" + str(img_i) + ".png")

    aps = evaluate_detections(all_boxes, all_gt_infos, classes, apply_mms=True, use_07_metric=True)
    ap_list = []
    for class_name in classes:
        if class_name == "__background__": continue
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.6f}'.format(class_name, aps[class_name]))
    print('Mean AP = {:.6f}'.format(np.nanmean(ap_list)))

    return aps

if __name__ == "__main__":

    model = load_model(os.path.join(".",  r"YOLOv2-Res101_reorg_bypass.model"))
    data_input = logging.graph.find_by_name(model, "data")
    img_width = data_input.shape[2]
    img_height= data_input.shape[1]

    if DATA_SET == "Pascal_VOC_2007":
        # obj_min, obj_max=1, 0
        # data_path= r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages"
        # img_list = [18,118,1118,27,2118,4118,1,2,3,4,5,6,7,8,9,10]
        # # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
        # save_path = os.path.join(".", "outputdir", "results", "pvoc2007")
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        #
        # for i in range(len(img_list)):
        #     img_name =  "{:06}.jpg".format(img_list[i])
        #     img = load_image(os.path.join(data_path, img_name))
        #
        #     preds = predictions_for_image(img, model, img_width, img_height)
        #     preds_nms = do_nms(preds)
        #     #import ipdb;ipdb.set_trace()
        #     color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        #     draw_bb_on_image(color_image, preds_nms)
        #
        #     if i<0:
        #         plot_image(color_image)
        #
        #     out_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        #
        #     save_image(out_img, save_path, "bb_"+img_name)
        #
        #     import math
        #     objectnesses = preds[:,4]
        #     obj_min = np.minimum(obj_min, np.minimum.reduce(objectnesses))
        #     obj_max = np.maximum(obj_max, np.maximum.reduce(objectnesses))
        # print((obj_min,obj_max))

        dataset_path = os.path.join("..", "..", "DataSets", "Pascal", "mappings")
        img_file = os.path.join(dataset_path, "test2007.txt")
        gtb_file = os.path.join(dataset_path, "test2007_rois_rel-ctr-wh_noPad_skipDif.txt")
        num_images_to_eval = 4952
        eval_map(model, img_file, gtb_file, num_images_to_eval)
# =======
# if __name__ == "__main__":
#
#     model = load_model(os.path.join(".", "outputdir", r"YOLOv2.model"))
#     data_input = logging.graph.find_by_name(model, "data")
#     img_width = data_input.shape[2]
#     img_height= data_input.shape[1]
#
#     if DATA_SET == "Pascal_VOC_2007":
#         obj_min, obj_max=1, 0
#         data_path= r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\JPEGImages"
#         img_list = [18,118,1118,27,2118,4118,1,2,3,4,5,6,7,8,9,10]
#         # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
#         save_path = os.path.join(".", "outputdir", "results", "pvoc2007")
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#
#         for i in range(len(img_list)):
#             img_name =  "{:06}.jpg".format(img_list[i])
#             img = load_image(os.path.join(data_path, img_name))
#
#             preds = predictions_for_image(img, model, img_width, img_height)
#             preds_nms = do_nms(preds)
#             #import ipdb;ipdb.set_trace()
#             color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#             draw_bb_on_image(color_image, preds_nms)
#
#             if i<0:
#                 plot_image(color_image)
#
#             out_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
#
#             save_image(out_img, save_path, "bb_"+img_name)
#
#             import math
#             objectnesses = preds[:,4]
#             obj_min = np.minimum(obj_min, np.minimum.reduce(objectnesses))
#             obj_max = np.maximum(obj_max, np.maximum.reduce(objectnesses))
#         print((obj_min,obj_max))
# >>>>>>> af7fe47dc9125065a809b34bf4764a60b0d52c35


    elif DATA_SET == "Grocery":
        obj_min, obj_max = 1, 0
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

            objectnesses = preds[:, 4]
            obj_min = np.minimum(obj_min, np.minimum.reduce(objectnesses))
            obj_max = np.maximum(obj_max, np.maximum.reduce(objectnesses))
        print((obj_min, obj_max))

    elif DATA_SET == "Overfit":
        obj_min, obj_max = 1, 0
        data_path = r"..\..\DataSets\Overfit"
        img_list = [37,38]
        # img_list = open(r"..\..\DataSets\Pascal\VOCdevkit\VOC2007\ImageSets\Main\test.txt").read().split()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputdir", "results", "overfit")
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