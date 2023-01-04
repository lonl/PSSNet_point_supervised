#Testing the output for same image input ,same Model predict.

import argparse
import sys,cv2
import ctypes
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import experiments, train, test, summary
from timer import Timer
from models import model_dict

lib_input_op = ctypes.cdll.LoadLibrary
lib_so = lib_input_op("./prediction_centroid/prediction_centroid.so")

ImgPath_input = "./datasets/test_data/12Months_Crop.jpg"
ImgPath = bytes(ImgPath_input.encode('utf-8'))

centroid_file = './datasets/test_data/ResUnet_100epoch/12Months_Crop_prediction_centroid_raw.txt'

# ----------------------------:Comb-Fusion----------------------------------
# raw_prediction_centroid file 2 fused_index file
test_file = centroid_file
raw_index = bytes(test_file.encode('utf-8'))
img_input = ImgPath

timer = Timer()
timer.tic()

dis_comb = 40
fused_index_ = ImgPath_input[:-4] + "_prediction_index_fused_dis_comb_{}.txt".format(dis_comb)
fused_index = bytes(fused_index_.encode('utf-8'))
img_output_ = ImgPath_input[:-4] + "_prediction_dis_comb_{}.jpg".format(dis_comb)
img_output = bytes(img_output_.encode('utf-8'))


lib_so.comb_xy(raw_index, img_input, fused_index, img_output, dis_comb)
print("FUSED INDEX DONE...")


timer.toc()
print(('Detection took {:.3f}s').format(timer.total_time))

# ----------------------------Compare-label-&-predicted-----------------------------------------------
#
dis_compare = 40
label_file = './datasets/test_data/12Months.txt'
label_file = bytes(label_file.encode('utf-8'))
index_fused_file = fused_index
lib_so.compare_center(label_file ,index_fused_file ,dis_compare)

print('ok ,compare finished')

