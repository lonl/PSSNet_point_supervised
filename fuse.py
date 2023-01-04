import argparse
import sys,cv2
import ctypes
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import os

lib_input_op = ctypes.cdll.LoadLibrary
lib_so = lib_input_op("./prediction_centroid/prediction_centroid.so")

# import argparse
# import experiments, train, test, summary
#
# lib_input_op = ctypes.cdll.LoadLibrary
# lib_so = lib_input_op("./prediction_centroid/prediction_centroid.so")


# ImgPath_input = "./datasets/download/acacia_data/Image_06months.jpg"
# centroid_file = "./figures/prediction_centroid_raw_txt"
# ImgPath = bytes(ImgPath_input.encode('utf-8'))
#
# raw_index = bytes(centroid_file.encode('utf-8'))
# img_input = ImgPath
#
# fused_index_ = ImgPath_input[:-4] + "_prediction_index_fused.txt"
# fused_index = bytes(fused_index_.encode('utf-8'))
# img_output_ = ImgPath_input[:-4] + "_prediction.jpg"
# img_output = bytes(img_output_.encode('utf-8'))
# dis_threshold = 30
#
# lib_so.comb_xy(raw_index, img_input, fused_index, img_output, dis_threshold)
# print("FUSED INDEX DONE...")



#check file
file_path1 = "./figures/IMG_0029_prediction_index_fused.txt"
centroid_file= "/mnt/a409/users/tongpinmo/projects/oilpalm/LCFCN/datasets/download/acacia_data/Image_06months_prediction_centroid_raw_txt"
ImgPath_input = "./datasets/download/acacia_data/Image_06months.jpg"
ImgPath = bytes(ImgPath_input.encode('utf-8'))

raw_index = bytes(centroid_file.encode('utf-8'))
img_input = ImgPath

fused_index_ = ImgPath_input[:-4] + "_prediction_index_fused.txt"
fused_index = bytes(fused_index_.encode('utf-8'))
img_output_ = ImgPath_input[:-4] + "_prediction.jpg"
img_output = bytes(img_output_.encode('utf-8'))
dis_threshold = 40

lib_so.comb_xy(raw_index, img_input, fused_index, img_output, dis_threshold)

