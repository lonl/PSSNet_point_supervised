#include "prediction_utils.hpp"

int count_lines(char *filename)
{
    std::ifstream ReadFile;
    int n = 0;
    std::string tmp;
    ReadFile.open(filename, std::ios::in); // read only

    while (getline(ReadFile, tmp, '\n')) {
        n++;
    }
    ReadFile.close();
    return n;
}



int cut_img(char *ImgPath, char *ResName_,
            int img_size_h, int img_size_w, int offset)
{

    cv::Mat Img = cv::imread(ImgPath);
    if (!Img.data) { cout << "can't open img" << endl; return -1; }
    cout << "open IMG Done." << endl;

    int h = Img.rows, w = Img.cols;

    int index_i = 0;
    for(int i = 0; i < (h - offset)/double(img_size_h - offset); i++){
        if(index_i + img_size_h > h)
        {
            index_i = h - img_size_h;
            if(index_i < 0)
            {
                index_i = 0;
                img_size_h = h;
            }
        }


        int index_j = 0;
        for(int j = 0; j < (w - offset)/double(img_size_w - offset); j++)
        {
            if(index_j + img_size_w > w)
            {
                index_j = w - img_size_w;
                if(index_j < 0)
                {
                    index_j = 0;
                    img_size_w = w;
                }
            }

            cv::Mat SubImg = cv::Mat::zeros(img_size_h, img_size_w, CV_8UC3);
            for(int m = 0; m < img_size_h; m++){
                for(int n = 0; n < img_size_w; n++){
                    SubImg.at<Vec3b>(m, n) = Img.at<Vec3b>(index_i + m, index_j + n);
                }
            }

            std::string ResName = ResName_;
            stringstream s1; s1 << index_i; ResName += s1.str();
            ResName += "_";
            stringstream s2; s2 << index_j; ResName += s2.str();
            ResName += ".jpg";
            cv::imwrite(ResName, SubImg);

            index_j = index_j + (img_size_w - offset);
        }

        index_i = index_i + (img_size_h - offset);

    }

    return 0;
}



int comb_xy(char *file_input_labels, char *file_input_img,
            char *file_output_labels, char *file_output_img, int dis_comb)
{
    cv::Mat input_img = cv::imread(file_input_img);
    int nline = count_lines(file_input_labels);
    ifstream infile_labels(file_input_labels);          //prediction_index_raw.txt
    ofstream outfile_labels(file_output_labels);        //index_fused.txt

    long int **palmxy = new long int *[nline];

    cout << "Input raw index number: " << nline << endl;
    for(int i = 0; i < nline; i++)
    {
        palmxy[i] = new long int[5];
        infile_labels >> palmxy[i][0];
        infile_labels >> palmxy[i][1];
        infile_labels >> palmxy[i][2];
        infile_labels >> palmxy[i][3];
        palmxy[i][4] = true;
    }

    for(int i = 0; i < nline; i++)
    {
        long int **xy_tmp = new long int*[nline];
        int count = 0;
        xy_tmp[count] = new long int[4];
        xy_tmp[count][0] = palmxy[i][0];
        xy_tmp[count][1] = palmxy[i][1];
        xy_tmp[count][2] = palmxy[i][2];
        xy_tmp[count][3] = palmxy[i][3];
        count++;

        if(palmxy[i][4]){
            for(int j = i + 1; j < nline; j++){
                long int x1 = (long int)(palmxy[i][0] + palmxy[i][2]) / 2;
                long int y1 = (long int)(palmxy[i][1] + palmxy[i][3]) / 2;
                long int x2 = (long int)(palmxy[j][0] + palmxy[j][2]) / 2;
                long int y2 = (long int)(palmxy[j][1] + palmxy[j][3]) / 2;

                long int dis = pow((x1 - x2), 2) + pow((y1 - y2), 2);
                dis = (long int)sqrt(dis);
                if(dis < dis_comb){
                    xy_tmp[count] = new long int[4];
                    xy_tmp[count][0] = palmxy[j][0];
                    xy_tmp[count][1] = palmxy[j][1];
                    xy_tmp[count][2] = palmxy[j][2];
                    xy_tmp[count][3] = palmxy[j][3];
                    count++;
                    palmxy[j][4] = false;
                }
            }

            long int sum_tmp_x1 = 0, sum_tmp_y1 = 0, sum_tmp_x2 = 0, sum_tmp_y2 = 0;
            for (int k = 0; k<count; k++) {
                sum_tmp_x1 += xy_tmp[k][0];
                sum_tmp_y1 += xy_tmp[k][1];
                sum_tmp_x2 += xy_tmp[k][2];
                sum_tmp_y2 += xy_tmp[k][3];
            }

            sum_tmp_x1 /= count;
            sum_tmp_x2 /= count;
            sum_tmp_y1 /= count;
            sum_tmp_y2 /= count;

            outfile_labels << sum_tmp_x1 << " " << sum_tmp_y1 << " "
                           << sum_tmp_x2 << " " << sum_tmp_y2 << endl;

            // plot the boounding boxes in the original pictures
            cv::rectangle(input_img, cv::Point(sum_tmp_x1, sum_tmp_y1), \
                          cv::Point(sum_tmp_x2, sum_tmp_y2), Scalar(0, 0, 255),1,8,0);

        }

    }
    //save the image to particular file
    cv::imwrite(file_output_img, input_img);

    return 0;
}


double compute_iou(double rec1[],double rec2[]){

    //computing IoU
    //:param rec1--ground-truth: (x0,y0,x1,y1), which reflects
    //:param rec2--predict-bbox: (x0, y0, x1, y1)
    //:return: scala value of IoU

   // computing area of each rectangles
    double S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1]);
    double S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1]);

   //computing the sum_area
    double sum_area = S_rec1 + S_rec2;

   // find the each edge of intersect rectangle
    double left_line = max(rec1[0], rec2[0]);
    double right_line = min(rec1[2], rec2[2]);
    double top_line = max(rec1[1], rec2[1]);
    double bottom_line = min(rec1[3], rec2[3]);

    // judge if there is an intersect
    if (left_line >= right_line || top_line >= bottom_line)
        return 0;

    else{
        double intersect = (right_line - left_line) * (bottom_line - top_line);
        double iou = (intersect / (sum_area - intersect) ) *1.0;

        std::cout << rec1[0] <<" " << rec1[1] <<" " << rec1[2] <<" " << rec1[3] << std::endl;
        std::cout << rec2[0] <<" " << rec2[1] <<" " << rec2[2] <<" " << rec2[3] << std::endl;
        std::cout << "IOU: " << iou << std::endl;

        return iou;
    }
}


double compare_xy(char *label_file,char *predicted_file)
{
    char *xy1 = label_file;
    char *xy2 = predicted_file;
    ifstream infile1(xy1);//labeled file
    ifstream infile2(xy2);//predicted file
    int nline1 = count_lines(xy1);
    int nline2 = count_lines(xy2);
    double **array1 = new double *[nline1];
    double **array2 = new double *[nline2];

    int count_1 = 0;
    int count_2 = 0;
    for(int i=0;i<nline1;i++){
        array1[i] = new double [5];
        infile1 >> array1[i][4];
        infile1 >> array1[i][1];
        infile1 >> array1[i][0];
        infile1 >> array1[i][3];
        infile1 >> array1[i][2];

        count_1++;
    }
    for(int i=0;i<nline2;i++){
        array2[i] = new double [4];
        infile2 >> array2[i][0];
        infile2 >> array2[i][1];
        infile2 >> array2[i][2];
        infile2 >> array2[i][3];
        count_2++;
    }


    double iou  = 0;
    double iou_sum = 0;
    double iou_avg = 0;


    for(int i=0;i<nline1;i++){  //labeled file
        double iou_final = 0;
        for(int j=0;j<nline2;j++){  //predicted file
            iou = compute_iou(array1[i],array2[j]);
            //cout << "iou= "<<iou<<endl;

            if (iou > iou_final){
                iou_final = iou;
                //continue;
            }
        }
        iou_sum += iou_final;

     }

    iou_avg  = (double) iou_sum / count_1 ;

    cout << "labeled= "<< count_1 <<" predicted="<< count_2 <<endl;
    cout << "iou_avg = "<< iou_avg << endl;

    return 0;
}


