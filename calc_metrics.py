
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
print("WARNINGS SUPRESSED")
warnings.filterwarnings("ignore")
import denoise_micrographs
from glob import glob
import pandas as pd
import os
import csv
import cv2
import sys
import random
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch
from tqdm import tqdm
import util.misc as utils

from models import build_model
from datasets.micrograph import make_micrograph_transforms

import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans

def nms(bounding_boxes, confidence_scores,feature_vectors, nms_threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_feats = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_scores[index])
        picked_feats.append(feature_vectors[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < nms_threshold)
        order = order[left]
    dummy = np.array(picked_boxes)
    picked_boxes = np.array(picked_boxes).squeeze()
    picked_score = np.array(picked_score)
    picked_feats = np.array(picked_feats)

    return picked_boxes, picked_score,picked_feats


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                            img_w, img_h
                            ], dtype=torch.float32)
    return b

#changes by Ashwin
def get_images(in_path):  
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' or ext == '.mrc':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Test hyperparameters
    parser.add_argument('--quartile_threshold', type=float, default=0.25, help='Quartile threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.7, help='Non-maximum suppression threshold')
    parser.add_argument('--empiar',default=None, help='EMPIAR ID for prediction')
    parser.add_argument('--complete',default=False, help='EMPIAR ID for prediction')
    parser.add_argument('--remarks', default='CryoTransformer_predictions', help='Additional remarks')
    parser.add_argument('--du_particles', default='N', choices=['Y', 'N'], help='DU Particles (Y or N)')
    # parser.add_argument('--num_queries', type=int, default=600, help='Number of queries')
    parser.add_argument('--save_micrographs_with_encircled_proteins', default='Y', choices=['Y', 'N'], help='Plot predicted proteins on Micrographs (Y or N)')
    parser.add_argument('--resume', default='pretrained_model/CryoTransformer_pretrained_model.pth', help='Resume path')
  

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=600, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    #  Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--contrastive_loss_coef', default=5, type=float)
    parser.add_argument('--clustering_loss_coef', default=2, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='micrograph')
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')


    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--thresh', default=0, type=float)   #edits by Ashwin, initially 0.5

    return parser


def resize_box_to_fixed_dimensions(bbox, new_width, new_height):
    # Calculate the center of the original box
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Calculate the new top-left and bottom-right coordinates
    top_left = (int(center_x - new_width / 2), int(center_y - new_height / 2))
    bottom_right = (int(center_x + new_width / 2), int(center_y + new_height / 2))

    return top_left, bottom_right

def boxes_to_mask_pred(boxes, image_size, target_size=(162, 162)):
    """
    Convert bounding boxes to an image mask and resize each box around its center to the target size.
    
    Parameters:
    - boxes: List of bounding boxes, each box is a list of [xmin, ymin, xmax, ymax]
    - image_size: Tuple of the image size (height, width)
    - target_size: Tuple of the target size (height, width) for each bounding box
    
    Returns:
    - mask: Binary mask with the same size as the input image
    """
    mask = np.zeros(image_size, dtype=np.uint8)  # Create a blank mask with the given image size
    target_h, target_w = target_size
    
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        box_w, box_h = xmax - xmin, ymax - ymin
        center_x, center_y = xmin + box_w // 2, ymin + box_h // 2
        
        # Calculate new box coordinates
        new_xmin = max(center_x - target_w // 2, 0)
        new_ymin = max(center_y - target_h // 2, 0)
        new_xmax = min(center_x + target_w // 2, image_size[1])
        new_ymax = min(center_y + target_h // 2, image_size[0])
        
        mask[new_ymin:new_ymax, new_xmin:new_xmax] = 1  # Set the mask to 1 inside the new bounding box
    
    return mask
def boxes_to_mask(boxes, image_size):
    """
    Convert bounding boxes to an image mask.
    
    Parameters:
    - boxes: List of bounding boxes, each box is a list of [xmin, ymin, xmax, ymax]
    - image_size: Tuple of the image size (width, height)
    
    Returns:
    - mask: Binary mask with the same size as the input image
    """
    mask = np.zeros(image_size, dtype=np.uint8)  # Create a blank mask with the given image size
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        mask[ymin:ymax, xmin:xmax] = 1  # Set the mask to 1 inside the bounding box
    
    return mask

def met_from_masks(gt_mask,pred_mask):
    true_positive = np.sum(np.logical_and(gt_mask, pred_mask))
    false_positive = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
    false_negative = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))
    smooth = 0.001  
    precision = (true_positive  + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive  + smooth) / (true_positive + false_negative + smooth)

    return precision,recall

@torch.no_grad()
def infer(images_path,annotation_dir, model, device):
    model.eval()
    duration = 0

    prefix_file_name = "EMPIAR_{}_remarks_{}".format(
    args.empiar, args.remarks
    )
    precisions = []
    recalls = []
    TPs = []
    FPs = []
    nGTs = []
    precisions_masks = []
    recalls_masks = []
    for img_sample in tqdm(images_path):
        filename = os.path.basename(img_sample)[:-4] 
        # print(len(filename))
        extension = img_sample[-3:]
        #loading image if input is in jpg format
        if extension == 'jpg':
            orig_image = Image.open(img_sample)
            # rgb_image = Image.open(img_sample).convert("RGB")
            img_size = orig_image.size
            rgb_image = Image.new("RGB", img_size)
            rgb_image.putdata([(x,x,x) for x in orig_image.getdata()])
            w, h = rgb_image.size

        if extension == 'mrc':
            orig_image = denoise_micrographs.denoise(img_sample)
            h, w = orig_image.shape
            # Create a new 3D array with shape (height, width, 3)
            rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
            # Set the same intensity value for all three color channels at each pixel location
            for y in range(h):
                for x in range(w):
                    intensity = orig_image[y, x]
                    rgb_array[y, x, 0] = intensity  # Red channel
                    rgb_array[y, x, 1] = intensity  # Green channel
                    rgb_array[y, x, 2] = intensity  # Blue channel
            # Convert the NumPy array to a PIL Image
            rgb_image = Image.fromarray(rgb_array)

        transform = make_micrograph_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(rgb_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)
        


        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        feature_array = outputs["z"].cpu().detach().numpy()
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        probas2 = outputs['pred_logits'].sigmoid()
        topk_values, topk_indexes = torch.topk(probas2.view(outputs["pred_logits"].shape[0], -1), args.num_queries, dim=1)   #extreme important mention num queries
        scores = topk_values
        keep = scores[0] > np.quantile(scores, 0)  #This is what prevents from predicting ice patches as particles
        scores = scores[0, keep]

        feature_array = feature_array[0, keep]
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], rgb_image.size)
        probas = probas[keep].cpu().data.numpy()

        scores = scores.cpu().detach().numpy()
        temp_boxes = [resize_box_to_fixed_dimensions(box.cpu().data.numpy().astype(np.int32), 162, 162) for box in bboxes_scaled ]
        flattened_list = [ [item for sublist in element for item in sublist] for element in temp_boxes]
        bboxes_scaled = torch.tensor(flattened_list)

        boxes, scores, feats = nms(bboxes_scaled, scores,feature_array, nms_threshold=args.nms_threshold)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feats)
        cluster_labels = kmeans.labels_
        
        if len(bboxes_scaled) == 0:
            print("there are no particle in image")
            continue
        gt_boxes = get_gt_boxes(filename,annotain_dir)

        def ret_f1(precision, recall):
            if precision + recall == 0:
                return 0.0
            f1_score = 2 * (precision * recall) / (precision + recall)
            return f1_score
        
        cluster_mask = (cluster_labels == 0)
        pred_boxes = boxes[cluster_mask]
        pred_scores = scores[cluster_mask]
        grp1_avg_conf = pred_scores.mean()
        p0, r0,TP0,FP0 = calculate_precision_recall_new(pred_boxes, gt_boxes)

        cluster_mask = (cluster_labels == 1)
        pred_boxes = boxes[cluster_mask]
        pred_scores = scores[cluster_mask]
        grp2_avg_conf = pred_scores.mean()
        p1, r1,TP1,FP1 = calculate_precision_recall_new(pred_boxes, gt_boxes)

        p2, r2,TP2,FP2 = calculate_precision_recall_new(boxes, gt_boxes)
        overall_avg_conf = scores.mean()

        f1_0 = ret_f1(p0, r0)
        f1_1 = ret_f1(p1, r1)
        f1_2 = ret_f1(p2, r2)
        
        grp_diff = abs(grp1_avg_conf - grp2_avg_conf)
        confidence_diff_threshold = 0.50 * max(grp1_avg_conf, grp2_avg_conf)
        if grp_diff > confidence_diff_threshold:
            if grp1_avg_conf > grp2_avg_conf and grp1_avg_conf > overall_avg_conf:
                p, r, TP, FP = p0, r0, TP0, FP0
            elif grp2_avg_conf > overall_avg_conf:
                p, r, TP, FP = p1, r1, TP1, FP1
            else:
                p, r, TP, FP = p2, r2, TP2, FP2
        else:
            # If the difference is not significant, fall back to overall scores
            p, r, TP, FP = p2, r2, TP2, FP2
        # p, r, TP, FP = p2, r2, TP2, FP2

        # if f1_0 > f1_1 and f1_0 > f1_2:
        #     p, r, TP, FP = p0, r0, TP0, FP0
        # elif f1_1 > f1_2:
        #     p, r, TP, FP = p1, r1, TP1, FP1
        # else:
        #     p, r, TP, FP = p2, r2, TP2, FP2

        TPs.append(TP)
        FPs.append(FP)
        nGTs.append(len(gt_boxes))
        precisions.append(p)
        recalls.append(r)

        pred_mask = boxes_to_mask_pred(boxes, (w, h),target_size=(108,108))
        gt_mask = boxes_to_mask(gt_boxes, (w, h))
        pm,rm = met_from_masks(pred_mask,gt_mask)
        precisions_masks.append(pm)
        recalls_masks.append(rm)

    total_TP = np.sum(TPs)
    total_FP = np.sum(FPs)
    total_nGT = np.sum(nGTs)
    total_R = total_TP/total_nGT
    total_P = (total_TP)/(total_TP + total_FP)
    return total_P,total_R,np.mean(precisions),np.mean(recalls),np.mean(precisions_masks),np.mean(recalls_masks)

        
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area


# MODIFY WDITH AND HEIGHT
def calculate_precision_recall_new(pred_bboxes, true_bboxes, threshold=0.5):
    """
    Calculate precision and recall for object detection based on bounding boxes.
    """
    new_width = 162
    new_height = 162
    pred_bboxes = [resize_box_to_fixed_dimensions(box.cpu().data.numpy().astype(np.int32), new_width, new_height) for box in pred_bboxes ]
    true_bboxes = [resize_box_to_fixed_dimensions(box.astype(np.int32), new_width, new_height) for box in true_bboxes ]

    TP = 0
    FP = 0
    det = np.zeros(len(true_bboxes))

    for pred_box in pred_bboxes:
        iouMax = sys.float_info.min
        jmax = -1
        for j, true_box in enumerate(true_bboxes):
            if(det[j] == 1):
                continue
            pred_top_left, pred_bottom_right = pred_box
            true_top_left, true_bottom_right = true_box
            iou = calculate_iou((pred_top_left[0], pred_top_left[1], pred_bottom_right[0], pred_bottom_right[1]),
                                (true_top_left[0], true_top_left[1], true_bottom_right[0], true_bottom_right[1]))
            if iou > iouMax:
                iouMax = iou
                jmax = j


        if iouMax >= threshold:

            if(det[jmax] == 0):
                TP+=1
                det[jmax] = 1
            else:
                FP+=1
        else:
            FP+=1

    rec = TP / len(true_bboxes) if len(true_bboxes) > 0 else 0
    pre = TP / (TP + FP) if (TP + FP) > 0 else 0
    # print(pre,rec,TP,FP)
    return pre,rec,TP,FP



def get_gt_boxes(filename,annotain_dir):
    gt_csv_path = os.path.join(annotain_dir, filename + ".csv")
    df = pd.read_csv(gt_csv_path)[["X-Coordinate", "Y-Coordinate", "Diameter"]]
    df["x_top_left"] = df["X-Coordinate"] - df["Diameter"] / 2
    df["y_top_left"] = df["Y-Coordinate"] - df["Diameter"] / 2
    df["x_bottom_right"] = df["X-Coordinate"] + df["Diameter"] / 2
    df["y_bottom_right"] = df["Y-Coordinate"] + df["Diameter"] / 2
    df = df[["x_top_left", "y_top_left", "x_bottom_right", "y_bottom_right"]]
    return df.values


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CryoTransformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    from datetime import datetime
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    

    device = torch.device(args.device)
     
    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    EMPIAR_IDS = [10017,10081,10093,10345,10532,11056]
    # EMPIAR_IDS = [10081]
    results = []

    for emp_id in EMPIAR_IDS:
        print("Evaluating on EMPIAR ID: ", emp_id)
        data_path = "/home/shiva/particle_picking/test_data/{}/images".format(emp_id)
        annotain_dir = "/home/shiva/particle_picking/test_data_annotations/{}/ground_truth/particle_coordinates".format(emp_id)
        image_paths = get_images(data_path)
        avg_p, avg_r,micro_avg_p,micro_avg_r,mask_avg_p,mask_avg_r = infer(image_paths,annotain_dir, model, device)
        results.append((emp_id,avg_p, avg_r))
        print("(emp_id,avg_p, avg_r,micro_avg_p,micro_avg_r,mask_avg_p,mask_avg_r)")
        print((emp_id, avg_p, avg_r,micro_avg_p,micro_avg_r,mask_avg_p,mask_avg_r))

    total_p = sum(p for _, p, _ in results)
    total_r = sum(r for _, _, r in results)
    avg_p_overall = total_p / len(results)
    avg_r_overall = total_r / len(results)

    # Print the results as a table
    print(f"{'EMPIAR ID':<10} {'Precision':<10} {'Recall':<10}")
    for emp_id, avg_p, avg_r in results:
        print(f"{emp_id:<10} {avg_p:<10.4f} {avg_r:<10.4f}")

    print(f"{'Average':<10} {avg_p_overall:<10.4f} {avg_r_overall:<10.4f}")