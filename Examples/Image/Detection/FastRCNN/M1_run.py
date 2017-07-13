# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from A2_RunWithPyModel import create_mb_source, train_fast_rcnn, base_path
import os
from cntk import *

import hierarchical_classification_tool as HCT

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.map.map_helpers import evaluate_detections


use_real_gt_not_sel_search_as_gt_info = True
#gt_boxes_rel_center = use_real_gt_not_sel_search_as_gt_info
use_gtbs_as_preds_aka_fake_output = False

output_scale = (1000, 1000)
#output_scale = (1080, 1920)
#output_scale = (1200, 1200)
#output_scale = (500, 500)
#output_scale = (774, 980)

###### Value Ranges as read #######
#
# rois (selective search)
### x = [219, 774]
### y = [  0, 980]
#
# gts (from dataset)
### x = [0.142, 0.693]
### y = [0.106, 0.781]
### x_center = [0.219, 0.637]
### y_center = [0.177, 0.704]
### w = [0.041, 0.223]
### h = [0.047, 0.200]

img_list = [r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_11_28_42_Pro.jpg",
                    r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_11_42_36_Pro.jpg",
                    r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_11_46_03_Pro.jpg",
                    r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_11_48_26_Pro.jpg",
                    r"D:\local\CNTK-2-0-rc1\cntk\Examples\Image\DataSets\Grocery\testImages\WIN_20160803_12_37_07_Pro.jpg"]

def prepare_ground_truth_boxes(gtbs, relative_coord=False, centered_coords=False, scale_input=None):
    """
    Creates an object that can be passed as the parameter "all_gt_infos" to "evaluate_detections" in map_helpers
    Parameters
    ----------
    gtbs - arraylike of shape (nr_of_images, nr_of_boxes, cords+original_label) where nr_of_boxes may be a dynamic axis

    Returns
    -------
    Object for parameter "all_gt_infos"
    """
    #import ipdb;ipdb.set_trace()
    num_test_images = len(gtbs)
    classes = HCT.output_mapper.get_all_classes()  # list of classes with new labels and indexing # todo: check if __background__ is present!!!
    all_gt_infos = {key: [] for key in classes}
    for image_i in range(num_test_images):
        image_gtbs = np.copy(gtbs[image_i])
        coords = image_gtbs[:,0:4]
        original_labels = image_gtbs[:,-1:]
        #mapped_labels = map_labels(original_labels)

        #image_gtbs[:,0] = (image_gtbs[:,0] -7/32)*16/9 #*9/16+7/32
        #image_gtbs[:, 0]-=6/32
        #image_gtbs[:, 0]*=16/9
        #image_gtbs[:,1]+=.1
        #image_gtbs[:, 2] = image_gtbs[:, 2] * 16/9#9 / 16

        #img = load_image(img_list[image_i])
        #img =draw_bb_on_image(img, image_gtbs)
        #plot_image(img)
        # make absolute

        # make coords bounding
        if centered_coords and False:
            xy = coords[:, :2]
            wh_half = coords[:, 2:] / 2
            coords = np.concatenate([xy - wh_half, xy + wh_half], axis=1)

        coords[:, [0, 2]] *= 9 / 16
        coords[:, [0, 2]] += 7 / 32

        if relative_coord:
            coords*= scale_input + scale_input


        #coords[:, [1, 3]] *= 1252
        #coords[:, [1, 3]] +=  -74

        #import ipdb;ipdb.set_trace()
        #img = draw_bb_on_image(img, points_to_xywh(coords/1000))
        #plot_image(img)


        #if True:
        #    coords[:,[0,2]]*=9/16
        #    coords[:, [0, 2]] += 1000*7/32

        # def to_one_hot(label, size):
        #     one_hot = np.zeros((size,))
        #     one_hot[label]=1
        #     return one_hot

        all_gt_boxes = []
        for gtb_i in range(len(image_gtbs)):
            label = int(original_labels[gtb_i][0])
            train_vector, _ = HCT.get_vectors_for_label_nr(label)
            reduced_vector = output_mapper.get_prediciton_vector(train_vector)  # remove lower backgrounds

            original_cls_name = HCT.cls_maps[0].getClass(label)
            for vector_i in range(1,len(reduced_vector)):
                if reduced_vector[vector_i]==0 : continue
                # else this label (vector_i) is active (either original or hypernym)

                current_class_name = classes[vector_i]
                if original_cls_name == current_class_name: original_cls_name = None

                lbox = np.concatenate([coords[gtb_i], [vector_i]], axis=0)
                lbox.shape=(1,)+lbox.shape
                all_gt_boxes.append(lbox)

            assert original_cls_name is None, "Original class label is not contained in mapped selection!"
            #if not original_cls_name is None: import ipdb;ipdb.set_trace()

        all_gt_boxes = np.concatenate(all_gt_boxes, axis=0)
        print("---all_gt_boxes.shape")
        print(all_gt_boxes.shape)



        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            #   gtBoxes = [box for box, label in zip(gtBoxes, gtLabels) if
            #              label.decode('utf-8') == self.classes[classIndex]]
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            #   gtInfos.append({'bbox': np.array(gtBoxes),
            #                   'difficult': [False] * len(gtBoxes),
            #                   'det': [False] * len(gtBoxes)})
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})


    return all_gt_infos


def prepare_predictions(outputs, roiss, num_classes):
    """

    Returns
    -------

    """
    num_test_images = len(outputs)

    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(num_classes)]

    for img_i in range(num_test_images):
        output = outputs[img_i]
        output.shape = output.shape[1:]
        rois = roiss[img_i]

        preds_for_img = []
        for roi_i in range(len(output)):
            pred_vector = output[roi_i]
            roi = rois[roi_i]

            processesed_vector = HCT.top_down_eval(pred_vector)
            reduced_p_vector = HCT.output_mapper.get_prediciton_vector(processesed_vector)

            assert len(reduced_p_vector)==num_classes
            for label_i in range(num_classes):
                if(reduced_p_vector[label_i]==0): continue
                prediciton = np.concatenate([roi, [reduced_p_vector[label_i], label_i]]) # coords+score+label
                prediciton.shape = (1,)+prediciton.shape
                preds_for_img.append(prediciton)

        preds_for_img = np.concatenate(preds_for_img, axis=0) # (nr_of_rois x 6) --> coords_scor_label

        for cls_j in range(1, num_classes):
            coords_score_label_for_cls = preds_for_img[np.where(preds_for_img[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

    return all_boxes






def eval_fast_rcnn_mAP(eval_model, img_map_file=None, roi_map_file=None):
    output_mapper = HCT.tree_map.get_output_mapper()
    classes = output_mapper.get_all_classes()
    num_test_images = 5
    num_classes = len(classes)
    num_original_classes=17
    num_channels=3
    image_height=1000#980
    image_width=1000#774
    feature_node_name='data'
    rois_per_image=2000
    gts_per_image = 20

    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()])#, name=feature_node_name)
    #roi_input = input_variable((rois_per_image, 5), dynamic_axes=[Axis.default_batch_axis()])
    roi_input = input_variable((rois_per_image, 4), dynamic_axes=[Axis.default_batch_axis()])
    #dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    label_input = input_variable((rois_per_image, num_classes))
    gt_input = input_variable((100))
    frcn_eval = eval_model(image_input, roi_input)

    if False:
        # Create the minibatch source
        minibatch_source = ObjectDetectionMinibatchSource(
            img_map_file, roi_map_file,
            max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
            pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
            randomize=False, use_flipping=False,
            max_images=cfg["CNTK"].NUM_TEST_IMAGES)

        # define mapping from reader streams to network inputs
        input_map = {
            minibatch_source.image_si: image_input,
            minibatch_source.roi_si: roi_input,
            minibatch_source.dims_si: dims_input
        }
    else:
        data_path=base_path
        data_set="test"
        minibatch_source = create_mb_source(image_height, image_width, num_channels,num_original_classes, rois_per_image, data_path, data_set)
        input_map = {
            minibatch_source.streams.features: image_input,
            minibatch_source.streams.rois: roi_input,
            minibatch_source.streams.roiLabels: label_input
        }
        input_map = {# add real gtb
            image_input: minibatch_source.streams.features,
            roi_input: minibatch_source.streams.rois,
            label_input: minibatch_source.streams.roiLabels,
            gt_input: minibatch_source.streams.gts
        }

    #img_key = cfg["CNTK"].FEATURE_NODE_NAME
    #roi_key = "x 5]"
    #dims_key = "[6]"


    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(num_classes)]

    all_raw_gt_boxes=[]
    all_raw_outputs=[]
    all_raw_rois=[]


    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    print(type(classes))
#    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        #import ipdb;ipdb.set_trace()
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)
 #       roi_data = mb_data[roi_input].asarray()
 #       roi_data.shape=(2000,4)
 #       print("--rois min, max")
 #       print(np.minimum.reduce(roi_data, axis=0))
 #       print(np.maximum.reduce(roi_data, axis=0))

        if use_real_gt_not_sel_search_as_gt_info:
            # receives rel coords
            gt_data = mb_data[gt_input].asarray()
            gt_data.shape = (gts_per_image, 5)
            all_gt_boxes=gt_data[np.where(gt_data[:,4]!=0)] # remove padded boxes!
        else :
            #receives abs coords
            gt_row = mb_data[roi_input].asarray()
            gt_row = gt_row.reshape((rois_per_image, 4))
            lbl_row= mb_data[label_input].asarray()
            lbl_row.shape = (rois_per_image, num_original_classes)
            lbl_clmn = np.argmax(lbl_row, axis=1)
            lbl_clmn.shape+=(1,)
            gt_row = np.concatenate([gt_row, lbl_clmn], axis=1)
            all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

        all_raw_gt_boxes.append(all_gt_boxes.copy())

            #           if gt_boxes_rel_center:
 #               #from rel_center to abs_edge
 #               #make abs
 #               all_gt_boxes[:,[0, 2]] *= 1000
 #               all_gt_boxes[:,[1, 3]] *= 1000
 #               #make edge
 #               w=all_gt_boxes[:,2]
 #               h=all_gt_boxes[:,3]
 #               x_c=all_gt_boxes[:,0]
 #               y_c = all_gt_boxes[:, 1]
#
 #               all_gt_boxes[:, 0] = x_c - w / 2
 #               all_gt_boxes[:, 1] = y_c - h / 2
 #               all_gt_boxes[:, 2] = x_c + w / 2
 #               all_gt_boxes[:, 3] = y_c + h / 2
 #           #all_gt_boxes = np.round(all_gt_boxes)
 #           all_gt_data.append(all_gt_boxes)  # TODO remove Debung code
 #       else:
 #           gt_row = mb_data[roi_input].asarray()
 #           gt_row = gt_row.reshape((rois_per_image, 4))
 #           lbl_row= mb_data[label_input].asarray()
 #           lbl_row.shape = (rois_per_image, num_original_classes)
 #           lbl_clmn = np.argmax(lbl_row, axis=1)
 #           lbl_clmn.shape+=(1,)
 #           gt_row = np.concatenate([gt_row, lbl_clmn], axis=1)
 #
 #           all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]


  #      for cls_index in range(num_original_classes):
  #          if cls_index == 0: continue
            #   gtBoxes = [box for box, label in zip(gtBoxes, gtLabels) if
            #              label.decode('utf-8') == self.classes[classIndex]]
  #          cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            #   gtInfos.append({'bbox': np.array(gtBoxes),
            #                   'difficult': [False] * len(gtBoxes),
            #                   'det': [False] * len(gtBoxes)})
  #          as_onehot=np.zeros((num_original_classes,))
  #          as_onehot[cls_index]=1
  #          train_vector,_ = HCT.get_vectors_for_label(as_onehot)
  #          reduced_vector = output_mapper.get_prediciton_vector(train_vector) # remove lower backgrounds
  #          assert np.add.reduce(reduced_vector)>0
            #cls_names=np.asarray(classes)[np.where(reduced_vector>0)]
            #import ipdb;ipdb.set_trace()

            # Append box for this class and all its hypernyms
            #for cls_name in cls_names:
  #          for i in np.where(reduced_vector>0)[0]:
  #              cls_name=classes[i]
                #if not cls_name == HCT.cls_maps[0].cls_map[i]: import ipdb;ipdb.set_trace()
  #              local_cls_boxes = np.concatenate([np.array(cls_gt_boxes)[:,:-1],np.ones((len(cls_gt_boxes),1))*i], axis=1)
                #if not np.alltrue(np.equal(local_cls_boxes, cls_gt_boxes)): import ipdb;ipdb.set_trace()
  #              all_gt_infos[cls_name].append({'bbox': local_cls_boxes,
  #                                         'difficult': [False] * len(cls_gt_boxes),
  #                                         'det': [False] * len(cls_gt_boxes)})

        #import ipdb;ipdb.set_trace()

        output = frcn_eval.eval({image_input: mb_data[image_input], roi_input: np.reshape(mb_data[roi_input].asarray(), roi_input.shape)})
        rois = mb_data[roi_input].asarray()
        rois.shape=(rois_per_image, 4)

        all_raw_rois.append(rois.copy())
        all_raw_outputs.append(output.copy())

        if False:
            def _process_output_slice(net_out, rois):

                assert net_out.shape[0] == rois.shape[0], print(net_out.shape, rois.shape)
                cords_score_label = []
                for box in range(rois_per_image):#len(net_out)
                    processed_vector = HCT.top_down_eval(net_out[box])
                    processed_vector_no_bg = output_mapper.get_prediciton_vector(processed_vector)
                    assert np.add.reduce(processed_vector_no_bg)>0
                    box_labels =  np.where(processed_vector_no_bg>0)[0]
                    box_scores = processed_vector_no_bg[box_labels]
                    #if len(box_labels) > 1 or box_labels[0]!=0: import ipdb;ipdb.set_trace()
                    if not len(box_labels>0): import ipdb;ipdb.set_trace()
                    # append the box for each label
                    for i in range(len(box_labels)):
                        cords_score_label.append(np.concatenate([rois[box], [box_scores[i], box_labels[i]]]))

                return np.asarray(cords_score_label)

            coords_score_label = _process_output_slice(output[0], rois)

            #fake output to be gtb!
            fake_coords = all_gt_boxes[:,0:4]
            fake_labels = all_gt_boxes[:,-1:]
            fake_score = np.ones(fake_labels.shape)
            fake_boxes = []
            for fb_i in range(len(fake_labels)):
                label = int(fake_labels[fb_i][0])
                train_vector, _ = HCT.get_vectors_for_label_nr(label)
                reduced_vector = output_mapper.get_prediciton_vector(train_vector)  # remove lower backgrounds

                original_cls_name = HCT.cls_maps[0].getClass(label)
                for vector_i in range(1, len(reduced_vector)):
                    if reduced_vector[vector_i] == 0: continue
                    # else this label (vector_i) is active (either original or hypernym)

                    current_class_name = classes[vector_i]
                    if original_cls_name == current_class_name: original_cls_name = None

                    lbox = np.concatenate([fake_coords[fb_i], [1, vector_i]], axis=0)
                    lbox.shape = (1,) + lbox.shape
                    fake_boxes.append(lbox)

                assert original_cls_name is None, "Original class label is not contained in mapped selection!"
            fake_boxes = np.concatenate(fake_boxes, axis=0)

            coords_score_label = fake_boxes


            #coords_score_label = np.concatenate([all_gt_boxes[:,0:4],np.ones((len(all_gt_boxes),1)),all_gt_boxes[:,4:5]], axis=1)
            print(coords_score_label.shape)

            #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
            for cls_j in range(1, num_classes):
                coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
                all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

            if (img_i+1) % 100 == 0:
                print("Processed {} samples".format(img_i+1))



    if use_gtbs_as_preds_aka_fake_output:
        # fake_coords=[]
        # fake_labels=[]
        #
        # for gtbs in all_raw_gt_boxes:
        #     #per img gtbs
        #     fake_box=[]
        #     orig_labels = gtbs[:,4]
        #     for label_i in len(gtbs):
        #         label = orig_labels[label_i]
        #
        #         train_vector, _ = HCT.get_vectors_for_label_nr(label)
        #         reduced_vector = output_mapper.get_prediciton_vector(train_vector)  # remove lower backgrounds
        #
        #         original_cls_name = HCT.cls_maps[0].getClass(label)
        #         for vector_i in range(1, len(reduced_vector)):
        #             if reduced_vector[vector_i] == 0: continue
        #             # else this label (vector_i) is active (either original or hypernym)
        #
        #             current_class_name = classes[vector_i]
        #             if original_cls_name == current_class_name: original_cls_name = None
        #
        #             lbox = np.concatenate([gtbs[label_i][:4], [vector_i]], axis=0)
        #             lbox.shape = (1,) + lbox.shape
        #             fake_box.append(lbox)
        #
        #         assert original_cls_name is None, "Original class label is not contained in mapped selection!"
        #     fake_box=np.concatenate(fake_box, axis=0)
        #     fake_coords.append(fake_box[:,:4])
        #     fake_labels.append(fake_box[:,4:])
        #
        # all_raw_outputs = fake_labels
        # all_raw_rois = fake_coords
        fake_coords=[]
        fake_outputs=[]

        for gtbs in all_raw_gt_boxes:
            coords = np.copy(gtbs[:,:4])

            # make absolute
            if use_real_gt_not_sel_search_as_gt_info:
                scale_input = output_scale
                coords *= scale_input + scale_input
                xy = coords[:, :2]
                wh_half = coords[:, 2:] / 2
                coords = np.concatenate([xy - wh_half, xy + wh_half], axis=1)

            fake_coords.append(coords)
            labels = gtbs[:,4]

            nr_of_labels = len(labels)
            fake_out = np.zeros((nr_of_labels, num_classes))
            for i in range(nr_of_labels):
                fake_out[i] ,_= HCT.get_vectors_for_label_nr(int(labels[i]))
            fake_out.shape = (1,) + fake_out.shape
            fake_outputs.append(fake_out)

        all_raw_outputs = fake_outputs
        all_raw_rois = fake_coords


    all_gt_infos = prepare_ground_truth_boxes(gtbs=all_raw_gt_boxes, relative_coord=use_real_gt_not_sel_search_as_gt_info, centered_coords=use_real_gt_not_sel_search_as_gt_info, scale_input=output_scale)
    all_boxes = prepare_predictions(all_raw_outputs, all_raw_rois, num_classes)

    visualize_gt(all_gt_infos)
    visualize_rois(all_boxes)

    aps = evaluate_detections(all_boxes, all_gt_infos, classes, apply_mms=False, use_07_metric=False)
    ap_list = []
    for class_name in classes:#sorted(aps):
        if class_name == "__background__":continue
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.6f}'.format(class_name, aps[class_name]))
    print('Mean AP = {:.6f}'.format(np.nanmean(ap_list)))

    import ipdb;
    ipdb.set_trace()
    return aps

def visualize_gt(all_gt_infos, plot=True):
    imgs = []
    for img_path in img_list:
        imgs.append(load_image(img_path))

    for cls_name in all_gt_infos:
        if cls_name == '__background__' : continue
        pred_list = all_gt_infos[cls_name]
        if not len(pred_list)== len(imgs): import ipdb;ipdb.set_trace()
        for img_i in range(len(imgs)):
            image = imgs[img_i]
            pred = np.copy(pred_list[img_i]["bbox"])

            add_rois_to_img(image, pred, cls_name)
            #if pred.size==0: continue

            #pred[:,0:4]/= output_scale+output_scale
            #pred[:,[0,2]]-=7/32
            #pred[:,[0,2]]*=16/9
            #draw_bb_on_image(image, points_to_xywh(pred), cls_name)
    if plot:
        for img in imgs:
            plot_image(img)

    return imgs

def visualize_rois(all_boxes, plot=True):
    classes = HCT.output_mapper.get_all_classes()
    imgs = []
    for img_path in img_list:
        imgs.append(load_image(img_path))

    for cls_i in range(len(all_boxes)):

        cls_name = classes[cls_i]
        for img_i in range(len(imgs)):
            image = imgs[img_i]
            rois = np.copy(all_boxes[cls_i][img_i])

            add_rois_to_img(image, rois, cls_name)
            #if rois.size==0: continue

            #rois[:,0:4] /= output_scale + output_scale
            #rois[:, [0, 2]] -= 7 / 32
            #rois[:,[0,2]] *= 16 / 9

            #import ipdb;ipdb.set_trace()
            #draw_bb_on_image(image, points_to_xywh(rois), cls_name)

    if plot:
        for img in imgs:
            plot_image(img)

    return imgs

def add_rois_to_img(img, rois, cls_name):
    if rois.size == 0: return

    rois[:, 0:4] /= output_scale + output_scale
    rois[:, [0, 2]] -= 7 / 32
    rois[:, [0, 2]] *= 16 / 9

    # import ipdb;ipdb.set_trace()
    draw_bb_on_image(img, points_to_xywh(rois), cls_name)

def draw_bb_on_image(image, bb_list, name=None):
    import cv2
    image_width = len(image[1])
    image_height = len(image)

    LIMIT_TO_FIRST = None
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
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

        if name is not None:
            cv2.putText(image, name, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 1, color,1)

    return image

def plot_image(image):
    import matplotlib.pyplot as mp
    mp.imshow(image)
    mp.plot()
    mp.show()

def load_image(img_path):
    import cv2
    return cv2.imread(img_path)

def points_to_xywh(points):
    xywh = np.zeros(points.shape)

    xywh[:, 0] = (points[:, 0] + points[:, 2]) / 2
    xywh[:, 1] = (points[:, 1] + points[:, 3]) / 2
    xywh[:, 2] = np.abs(points[:, 2] - points[:, 0])
    xywh[:, 3] = np.abs(points[:, 3] - points[:, 1])
    xywh[:, 4:] = points[:, 4:]

    return xywh


if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "Output", "frcn_py.model")

    # Train only is no model exists yet
    if os.path.exists(model_path):
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_fast_rcnn()
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    # eval trained_model
    print("\n---Evaluation---")

    output_mapper = HCT.tree_map.get_output_mapper()
    known_classes = output_mapper.get_all_classes()

    aps=eval_fast_rcnn_mAP(trained_model)

