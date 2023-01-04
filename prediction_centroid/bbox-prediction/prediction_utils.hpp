#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

extern "C" {
int count_lines(char *filename);
int cut_img(char *ImgPath, char *ResName, int img_size_h, int img_size_w, int offset);
int comb_xy(char *file_input_labels, char *file_input_img,
            char *file_output_labels, char *file_output_img, int dis_comb);
double compare_xy(char *label_file,char *predicted_file);
double compute_iou(double rec1[],double rec2[]);
}
