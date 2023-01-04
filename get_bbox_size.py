import cv2
import os
import numpy as np


bbox_file = './datasets/test_data/12months_predict_bbox.txt'


with open(bbox_file,'r') as tf:
    bbox_file = tf.readlines()
    w_sum = 0
    h_sum = 0
    count = 0
    for line in bbox_file:
        count += 1
        line = line.strip().split(' ')
        # print('line: ',line)
        x_0 = int(line[0])
        y_0 = int(line[1])
        x_1 = int(line[2])
        y_1 = int(line[3])

        w_sum += x_1 - x_0
        h_sum += y_1 - y_0
    print('line: ',count)
    width_avg  = w_sum / (count * 2)
    height_avg = h_sum / (count * 2 )

    print('width_avg: ',width_avg)
    print('height_avg: ',height_avg)



    



