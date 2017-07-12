import os,sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))

from utils.map.map_helpers import evaluate_detections
import numpy as np

num_classes = 6
classes=["__background__","a", "b", "c", "d", "e"]
gtb_input= np.asarray([[ 564.5,  135. ,  598. ,  191. ,   1. ],
       [ 523.5,  317. ,  599. ,  347. ,    2. ],
       [ 250. ,  234. ,  347. ,  292. ,    3. ],
       [ 584.5,  385. ,  637. ,  468. ,    4. ],
       [ 406.5,  497.5,  484. ,  545. ,    2. ],
       [ 280.5,  452.5,  341. ,  509. ,    2. ],
       [ 540.5,  587. ,  602. ,  684. ,    4. ],
       [ 449.5,  599.5,  500. ,  689. ,    3. ],
       [ 235. ,  586. ,  318. ,  674. ,    5. ]], dtype=np.float32)
num_test_images=1

def test_eval():
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(num_classes)]

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        gt_row = gtb_input
        all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

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

        regressed_rois = gtb_input[:,:4]
        scores = np.ones((len(gtb_input),1))
        labels = gtb_input[:,4:]
        coords_score_label = np.hstack((regressed_rois, scores, labels))

        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        for cls_j in range(1, num_classes):
            coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

        if (img_i+1) % 100 == 0:
            print("Processed {} samples".format(img_i+1))

    # calculate mAP
    aps = evaluate_detections(all_boxes, all_gt_infos, classes,
                              nms_threshold=.5,
                              conf_threshold =0)
    ap_list = []
    for class_name in aps:
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.4f}'.format(class_name, aps[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(ap_list)))
    assert np.add.reduce(ap_list)==num_classes-1


test_eval()
