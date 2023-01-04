import argparse
import sys,cv2
import ctypes
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import torch


lib_input_op = ctypes.cdll.LoadLibrary
lib_so = lib_input_op("./prediction_centroid/prediction_centroid.so")


ImgPath_input = "./datasets/download/acacia_data/Image_12months.jpg"

dis_comb = 30
fused_index_ = ImgPath_input[:-4] + "_prediction_index_fused_dis_comb_{}.txt".format(dis_comb)
fused_index = bytes(fused_index_.encode('utf-8'))   #

dis_compare = 20
label_file = 'datasets/download/acacia_data/12Months.txt'
label_file = bytes(label_file.encode('utf-8'))
index_fused_file = fused_index
lib_so.compare_center(label_file, index_fused_file, dis_compare)
#

print('ok ,compare finished')

