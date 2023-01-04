import argparse
import sys,cv2
import ctypes
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import torch
import argparse


#--------------------------------------first step:comb bbox-------------
lib_input_op = ctypes.cdll.LoadLibrary
lib_so = lib_input_op("./prediction_centroid/bbox-prediction/prediction_utils.so")

ImgPath_input = "./datasets/test_data/12Months_Crop.jpg"
ImgPath = bytes(ImgPath_input.encode('utf-8'))

centroid_file = './datasets/test_data/mask2unet-bbox/12months_mask2unet_bbox.txt'

# ----------------------------:Comb-Fusion---- ------------------------------
# raw_prediction_centroid file 2 fused_index file
test_file = centroid_file
raw_index = bytes(test_file.encode('utf-8'))
img_input = ImgPath

dis_comb = 40
fused_index_ = ImgPath_input[:-4] + "_prediction_bbox_fused_dis_comb_{}.txt".format(dis_comb)
fused_index = bytes(fused_index_.encode('utf-8'))
img_output_ = ImgPath_input[:-4] + "_prediction_dis_comb_{}.jpg".format(dis_comb)
img_output = bytes(img_output_.encode('utf-8'))


lib_so.comb_xy(raw_index, img_input, fused_index, img_output, dis_comb)
print("FUSED INDEX DONE...")


#---------------------------------------second step:IOU------------------
# fused_index = centroid_file
# fused_index = bytes(fused_index.encode('utf-8'))
label_file = './datasets/test_data/12Months_bb.txt'
label_file = bytes(label_file.encode('utf-8'))
index_fused_file = fused_index


lib_so.compare_xy.restype = ctypes.c_float
lib_so.compare_xy(label_file, index_fused_file)

print('ok ,compare finished')


# --------------------------------calculate iou-------------------------
# def compute_iou(rec1, rec2):
#     """
#     computing IoU
#     :param rec1: (x0, y0, x1, y1), which reflects
#             (top, left, bottom, right)
#     :param rec2: (x0, y0, x1, y1)
#     :return: scala value of IoU
#     """
#     # computing area of each rectangles
#     S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
#     S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
#
#     # computing the sum_area
#     sum_area = S_rec1 + S_rec2
#     print('sum_are: ',sum_area)
#
#     # find the each edge of intersect rectangle
#     left_line = max(rec1[0], rec2[0])
#     right_line = min(rec1[2], rec2[2])
#     top_line = max(rec1[1], rec2[1])
#     bottom_line = min(rec1[3], rec2[3])
#
#     # judge if there is an intersect
#     if left_line >= right_line or top_line >= bottom_line:
#         return 0
#     else:
#         intersect = (right_line - left_line) * (bottom_line - top_line)
#         print('intersect: ',intersect)
#         return (intersect / (sum_area - intersect)) * 1.0
#
#
# if __name__ == '__main__':
#
#     rect1 = (27, 661, 47,679)
#     # (top, left, bottom, right)
#     rect2 = (28,662,46,678)
#     iou = compute_iou(rect1, rect2)
#     print(iou)