
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from sklearn.cluster import KMeans
import torch

import util.misc as utils

from models import build_model
from datasets.micrograph import make_micrograph_transforms

import matplotlib.pyplot as plt
import time


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
    parser.add_argument('--empiar', default='10081', help='EMPIAR ID for prediction')
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



@torch.no_grad()
def infer(images_path,annotain_dir, model, postprocessors, device, output_dir):
    model.eval()
    duration = 0

    prefix_file_name = "EMPIAR_{}_remarks_{}".format(
    args.empiar, args.remarks
    )

    for img_sample in images_path:
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


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        # hooks = [
        #     model.backbone[-2].register_forward_hook(
        #                 lambda self, input, output: conv_features.append(output)

        #     ),
        #     model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #                 lambda self, input, output: enc_attn_weights.append(output[1])

        #     ),
        #     model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
        #                 lambda self, input, output: dec_attn_weights.append(output[1])

        #     ),

        # ]
        print(image.shape)
        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        feature_array = outputs["z"].cpu().detach().numpy()
        # print(outputs["pred_logits"])
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # print("=============probas softmax ===============================")
        # print(probas)

        probas2 = outputs['pred_logits'].sigmoid()
        topk_values, topk_indexes = torch.topk(probas2.view(outputs["pred_logits"].shape[0], -1), args.num_queries, dim=1)   #extreme important mention num queries
        scores = topk_values
        keep = scores[0] > np.quantile(scores, args.quartile_threshold)  #This is what prevents from predicting ice patches as particles
        scores = scores[0, keep]


        # keep = probas.max(-1).values > args.thresh  #this is original
        # print("==========" + img_sample + "====pred_logits after softmax===============================")
        # print(keep )
        feature_array = feature_array[0, keep]
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], rgb_image.size)
        probas = probas[keep].cpu().data.numpy()


        # for hook in hooks:
        #     hook.remove()

        # conv_features = conv_features[0]
        # enc_attn_weights = enc_attn_weights[0]
        # dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        # h, w = conv_features['0'].tensors.shape[-2:]
        scores = scores.cpu().detach().numpy()
        print("bbox_scaled shape: ",bboxes_scaled.shape)
        temp_boxes = [resize_box_to_fixed_dimensions(box.cpu().data.numpy().astype(np.int32), 162, 162) for box in bboxes_scaled ]
        flattened_list = [ [item for sublist in element for item in sublist] for element in temp_boxes]
        bboxes_scaled = torch.tensor(flattened_list)
        boxes, scores, feats = nms(bboxes_scaled, scores,feature_array, nms_threshold=args.nms_threshold)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feats)
        cluster_labels = kmeans.labels_
        gt_boxes = get_gt_boxes(filename,annotain_dir)


        overall_avg_conf = scores.mean()
        cluster_mask = (cluster_labels == 0)
        pred_boxes_1 = boxes[cluster_mask]
        pred_scores_1 = scores[cluster_mask]
        grp1_avg_conf = pred_scores_1.mean()

        cluster_mask = (cluster_labels == 1)
        pred_boxes_2 = boxes[cluster_mask]
        pred_scores_2 = scores[cluster_mask]
        grp2_avg_conf = pred_scores_2.mean()

        grp_diff = abs(grp1_avg_conf - grp2_avg_conf)
        confidence_diff_threshold = 0.50 * max(grp1_avg_conf, grp2_avg_conf)
        if grp_diff > confidence_diff_threshold:
            if grp1_avg_conf > grp2_avg_conf and grp1_avg_conf > overall_avg_conf:
                boxes = pred_boxes_1
            elif grp2_avg_conf > overall_avg_conf:
                boxes = pred_boxes_2
            else:
                boxes = boxes
        else:
            # If the difference is not significant, fall back to overall scores
            boxes = boxes

        print(f"----- generating star file for {filename}")
        # create directory for star files if not exist:
        box_file_path = output_dir + '/box_files/'
        predicted_particles_visualizations_path = output_dir + '/predicted_particles_visualizations/'
        if not os.path.exists(box_file_path):
            os.makedirs(box_file_path)
        if not os.path.exists(predicted_particles_visualizations_path):
            os.makedirs(predicted_particles_visualizations_path)
        save_individual_box_file(boxes, scores, img_sample, h, box_file_path, "_CryoTransformer")
        # print("=============boxes  ===============================")
        # print(boxes)
        # print("=============scores  ===============================")
        # print(scores)
        #edits by Ashwin
        if len(bboxes_scaled) == 0:
            print("there are no particle in image")
            continue
        if args.save_micrographs_with_encircled_proteins == 'Y':
            # plot_predicted_boxes(rgb_image, boxes, filename, predicted_particles_visualizations_path, h)
            plot_predicted_boxes_and_gt_boxes(rgb_image, boxes, gt_boxes, filename,predicted_particles_visualizations_path, h)

        # print("=============== Predictions saved ===================")
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # infer_time = end_t - start_t
        # duration += infer_time
        # print("Processing END...{} ({:.3f}s)".format(filename, infer_time))

    # avg_duration = duration / len(images_path)

    # print("Avg. Time: {:.3f}s".format(avg_duration))

    #making header for combined star file:
    save_combined_star_file(box_file_path, prefix_file_name)


def save_individual_box_file(boxes, scores, img_file, h, box_file_path, out_imgname):
    write_name = box_file_path + os.path.basename(img_file)[:-4] + out_imgname + '.box'
    with open(write_name, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow(["Micrograph_Name    X_Coordinate    Y_Coordinate    Class_Number    AnglePsi    Confidence_Score"])

        for i, box in enumerate(boxes):
            star_bbox = box.cpu().data.numpy()
            star_bbox = star_bbox.astype(np.int32)
            #h- is done to handle the cryoSparc micrograph reading orientation
            boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', (star_bbox[0] + star_bbox[2]) / 2, h-(star_bbox[1] + star_bbox[3]) / 2, -9999, -9999, scores[i]])
            if args.du_particles == 'Y':
                coordinate_shift_rand = random.choice(list(range(-20, -9)) + list(range(10, 21))) #shifting center to obtain better 2D averaging
                # coordinate_shift_rand = 10
                boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', ((star_bbox[0] + star_bbox[2]) / 2)+coordinate_shift_rand, (h-(star_bbox[1] + star_bbox[3]) / 2)+coordinate_shift_rand, -9999, -9999, scores[i]])

def plot_predicted_boxes(rgb_image, boxes, filename, predicted_particles_visualizations_path, h):
    img = np.array(rgb_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(boxes):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox_d = bbox.astype(np.int32)
        bbox_circle = bbox.astype(np.int32)


        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ])
        bbox = bbox.reshape((4, 2))
        # bbox_d = np.array([
        #     [bbox_d[0]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[3]+15],
        #     [bbox_d[0]+15, bbox_d[3]+15],
        #     ])
        # bbox_d = bbox_d.reshape((4, 2))


        bbox_circle_center = np.array([(bbox_circle[0] + bbox_circle[2]) / 2, (bbox_circle[1] + bbox_circle[3])/2]) #h- is ommitted here to handle the image plot
        bbox_circle_center = bbox_circle_center.reshape((1, 2))

        x_coordinate, y_coordinate = bbox_circle_center[0]
        center = (int(x_coordinate), int(y_coordinate))


        # cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
        # color=(0,255,0) #green
        color =(150, 255, 255) #purple
        radius=81
        thickness=10 # 7 earlier
        # cv2.polylines(img, [bbox_d], True, (0, 255, 0), 4)
        cv2.circle(img, center, radius, color, thickness)

    img_save_path = os.path.join(predicted_particles_visualizations_path, f"{filename}.jpg")

    cv2.imwrite(img_save_path, img)

def plot_predicted_boxes_and_gt_boxes(rgb_image, boxes, gt_boxes, filename,
                                      predicted_particles_visualizations_path,
                                      h):
    print("Plotting both predicted and gt boxes")
    img = np.array(rgb_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # for idx, box in enumerate(boxes):
    #     bbox = box.cpu().data.numpy()
    #     bbox = bbox.astype(np.int32)
    #     bbox_d = bbox.astype(np.int32)
    #     bbox_circle = bbox.astype(np.int32)

    #     bbox = np.array([
    #         [bbox[0], bbox[1]],
    #         [bbox[2], bbox[1]],
    #         [bbox[2], bbox[3]],
    #         [bbox[0], bbox[3]],
    #     ])
    #     bbox = bbox.reshape((4, 2))
    #     # bbox_d = np.array([
    #     #     [bbox_d[0]+15, bbox_d[1]+15],
    #     #     [bbox_d[2]+15, bbox_d[1]+15],
    #     #     [bbox_d[2]+15, bbox_d[3]+15],
    #     #     [bbox_d[0]+15, bbox_d[3]+15],
    #     #     ])
    #     # bbox_d = bbox_d.reshape((4, 2))

    #     bbox_circle_center = np.array([
    #         (bbox_circle[0] + bbox_circle[2]) / 2,
    #         (bbox_circle[1] + bbox_circle[3]) / 2
    #     ])  #h- is ommitted here to handle the image plot
    #     bbox_circle_center = bbox_circle_center.reshape((1, 2))

    #     x_coordinate, y_coordinate = bbox_circle_center[0]
    #     center = (int(x_coordinate), int(y_coordinate))

    #     # cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
    #     # color=(0,255,0) #green
    #     # color = (150, 255, 255)  #purple
    #     color = (0, 0, 255)
    #     radius = 81
    #     thickness = 10  # 7 earlier
    #     # cv2.polylines(img, [bbox_d], True, (0, 255, 0), 4)
    #     cv2.circle(img, center, radius, color, thickness)

    for idx, box in enumerate(gt_boxes):
        bbox = box  #.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox_d = bbox.astype(np.int32)
        bbox_circle = bbox.astype(np.int32)

        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ])
        bbox = bbox.reshape((4, 2))
        # bbox_d = np.array([
        #     [bbox_d[0]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[3]+15],
        #     [bbox_d[0]+15, bbox_d[3]+15],
        #     ])
        # bbox_d = bbox_d.reshape((4, 2))

        bbox_circle_center = np.array([
            (bbox_circle[0] + bbox_circle[2]) / 2,
            (bbox_circle[1] + bbox_circle[3]) / 2
        ])  #h- is ommitted here to handle the image plot
        bbox_circle_center = bbox_circle_center.reshape((1, 2))

        x_coordinate, y_coordinate = bbox_circle_center[0]
        center = (int(x_coordinate), int(y_coordinate))

        # cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
        color = (0, 255, 0)  #green
        # color = (150, 255, 255)  #purple
        radius = 79
        thickness = 10  # 7 earlier
        # cv2.polylines(img, [bbox_d], True, (0, 255, 0), 4)
        cv2.circle(img, center, radius, color, thickness)

    vio = (255, 0, 180)
    #---Color of the border---
    img_bordered = cv2.copyMakeBorder(img,
                                      500,
                                      0,
                                      0,
                                      0,
                                      cv2.BORDER_CONSTANT,
                                      value=vio)

    p, r = calculate_precision_recall_new(boxes, gt_boxes)
    BLACK = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 3
    font_color = BLACK
    font_thickness = 3
    text = 'Precision: {} | Recall: {} | pred_boxes: {} | gt_boxes: {}'.format(np.round(p, 2), np.round(r, 2),len(boxes),len(gt_boxes))
    print(text)
    x, y = 10, 650
    img_text = cv2.putText(img_bordered, text, (200, 200), font, font_size,
                           font_color, font_thickness, cv2.LINE_AA)

    img_save_path = os.path.join(predicted_particles_visualizations_path,
                                 f"{filename}.jpg")
    print("Saved file at : ", img_save_path)
    cv2.imwrite(img_save_path, img_text)

def save_combined_star_file(box_file_path, prefix_file_name):
    text_files = [file for file in os.listdir(box_file_path) if file.endswith('.box')]
    text_files.sort()
    output_file = output_dir + prefix_file_name + '_' + 'star_file.star'
    header = '''
data_

loop_
_rlnMicrographName #1 
_rlnCoordinateX #2 
_rlnCoordinateY #3 
_rlnClassNumber #4 
_rlnAnglePsi #5
_rlnAutopickFigureOfMerit #6
'''

    with open(output_file, 'w') as outfile:
        # Write the header content to the new file
        outfile.write(header)

        # Iterate over each text file
        for file in text_files:
            # Open the current file in read mode
            with open(os.path.join(box_file_path, file), 'r') as infile:
                # Skip the first line
                next(infile)
                # Read the remaining content of the file
                content = infile.read()
                # Write the content to the new file
                outfile.write(content)


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
    print(pre,rec,TP,FP,len(true_bboxes))
    return pre,rec

## Need to fix metric.. each true box can only have one matched pred box
def calculate_precision_recall(pred_bboxes, true_bboxes, threshold=0.5):
    """
    Calculate precision and recall for object detection based on bounding boxes.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    found_true_boxes = set()
    for pred_box in pred_bboxes:
        is_true_positive = False
        for i, true_box in enumerate(true_bboxes):
            iou = calculate_iou(pred_box, true_box)
            if iou >= threshold:
                found_true_boxes.add(i)
                true_positives += 1
                is_true_positive = True
                break
        if not is_true_positive:
            false_positives += 1
    false_negatives = len(true_bboxes) - len(found_true_boxes)
    # false_negatives = len(true_bboxes) - true_positives

    precision = true_positives / (
        true_positives +
        false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (len(true_bboxes)) if len(true_bboxes) != 0 else 0

    if (recall > 1):
        print("RECALL GREATER THAN 1")
        print("TP: {}, FN: {}".format(true_positives, false_negatives))
    return precision, recall



def get_gt_boxes(filename,annotain_dir):
    # particle_coords_path = "/home/shivasankaran/projects/particle_picking/train_val_test_data/particle_coordinates"
    # particle_coords_path = "/home/shivasankaran/projects/particle_picking/CryoTransformer/test_data_annotations/11056/ground_truth/particle_coordinates"
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
    # data_path = "test_data/{}/images".format(args.empiar) #cryoPPP ~300 micrographs /home/shivasankaran/projects/particle_picking/test_data
    # data_path = "/home/shivasankaran/projects/particle_picking/test_data/{}/images".format(args.empiar)
    data_path = "/home/shivasankaran/projects/particle_picking/train_val_test_data/test"
    
    EMPIAR_ID = 11056
    output_dir = "output/predictions/predictions_EMPIAR_{}_remarks_{}_timestamp_{}/".format(
    EMPIAR_ID, args.remarks, timestamp)

    # output_dir = "output/predictions/{}/".format(EMPIAR_ID)
    # checkpoint = torch.load("/home/shivasankaran/projects/particle_picking/Deformable-DETR/checkpoint.pth", map_location='cpu')
    # training_args = checkpoint["args"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    data_path = "/home/shiva/particle_picking/test_data/{}/images".format(EMPIAR_ID)
    annotain_dir = "/home/shiva/particle_picking/test_data_annotations/{}/ground_truth/particle_coordinates".format(EMPIAR_ID)
    image_paths = get_images(data_path)
    # image_paths = [
    # "/home/shivasankaran/projects/particle_picking/test_data/11056/images/ja_115-1_0000_May14_21.14.32.jpg",
    # "/home/shivasankaran/projects/particle_picking/test_data/10017/images/Falcon_2012_06_12-15_43_48_0.jpg",
    # "/home/shivasankaran/projects/particle_picking/test_data/10017/images/Falcon_2012_06_13-01_05_13_0.jpg",
    # "/home/shivasankaran/projects/particle_picking/test_data/10017/images/Falcon_2012_06_12-14_33_35_0.jpg",
    # "/home/shivasankaran/projects/particle_picking/test_data/10345/images/18jam15a_0007_ali_DW.jpg"]
    print(image_paths)

    infer(image_paths,annotain_dir, model, postprocessors, device, output_dir)