#include "prediction_centroid.hpp"

int count_lines(char *filename){
        ifstream ReadFile;
        int n = 0;
        string tmp;
        ReadFile.open(filename, ios::in);//ios::in read only

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
    ifstream infile_labels(file_input_labels);               //prediction_centroid_raw.txt
    ofstream outfile_labels(file_output_labels);             //index_fused.txt

    long int **palmxy = new long int *[nline];

    cout << "Input raw index number: " << nline << endl;
    for(int i = 0; i < nline; i++)
    {
        palmxy[i] = new long int[3];
        infile_labels >> palmxy[i][0];
        infile_labels >> palmxy[i][1];
        palmxy[i][2] = true;
    }

    for(int i = 0; i < nline; i++)
    {
        long int **xy_tmp = new long int*[nline];
        int count = 0;
        xy_tmp[count] = new long int[2];
        xy_tmp[count][0] = palmxy[i][0];
        xy_tmp[count][1] = palmxy[i][1];
        count++;

        if(palmxy[i][2])
        {
            for(int j = i + 1; j < nline; j++)
            {
                long int x1 = (long int)(palmxy[i][0]);
                long int y1 = (long int)(palmxy[i][1]);
                long int x2 = (long int)(palmxy[j][0]);
                long int y2 = (long int)(palmxy[j][1]);

                long int dis = pow((x1 - x2), 2) + pow((y1 - y2), 2);
                dis = (long int)sqrt(dis);
                if(dis < dis_comb)
                {
                    xy_tmp[count] = new long int[2];
                    xy_tmp[count][0] = palmxy[j][0];
                    xy_tmp[count][1] = palmxy[j][1];
                    count++;
                    palmxy[j][2] = false;
                }
            }

            long int sum_tmp_x1 = 0, sum_tmp_y1 = 0;
            for (int k = 0; k<count; k++)
            {
                sum_tmp_x1 += xy_tmp[k][0];
                sum_tmp_y1 += xy_tmp[k][1];
            }

            sum_tmp_x1 /= count;
            sum_tmp_y1 /= count;

            outfile_labels << sum_tmp_x1 << " " << sum_tmp_y1 << endl;

            // plot solid-circles in the original pictures
            cv::circle(input_img, cv::Point(sum_tmp_x1, sum_tmp_y1),8,cv::Scalar(0, 0, 255),-1,8,0);

        }

    }

    cv::imwrite(file_output_img, input_img);

    return 0;
}


int compare_center(char *label_file,char *fused_file,int dis_compare)
{
    char *xy1 = label_file;
    char *xy2 = fused_file;
    ifstream infile1(xy1);//labeled file
    ifstream infile2(xy2);//predicted file
    int nline1 = count_lines(xy1);
    cout << nline1 << endl;
    int nline2 = count_lines(xy2);
    cout << nline2 <<endl;
    double **array1 = new double *[nline1];
    double **array2 = new double *[nline2];
    int count_1 = 0;
    int count_2 = 0;
    for(int i=0;i<nline1;i++){
        array1[i] = new double [2];
        infile1 >> array1[i][0];
        infile1 >> array1[i][1];

        count_1++;
    }
    for(int i=0;i<nline2;i++){
        array2[i] = new double [2];
        infile2 >> array2[i][0];
        infile2 >> array2[i][1];
        count_2++;
    }

    int tn1=0,tn2=0,tn3=0,tn4=0,tn5=0,tn6=0;
    int count_true = 0;
    for(int i=0;i<nline1;i++){
        for(int j=0;j<nline2;j++){
            double x1 = (array1[i][0]) ;
            double y1 = (array1[i][1]) ;
            double x2 = (array2[j][0]) ;
            double y2 = (array2[j][1]) ;
            double dis = pow((x1 - x2), 2) + pow((y1 - y2), 2);
            dis = (double)sqrt(dis);
            if(dis < dis_compare ){
                count_true++;
                continue;
            }
        }
    }
    double recall    = (double)count_true / count_1 ;
    double precision = (double) count_true / count_2;
    double f_score   = (double) 2 / ((1/precision) + (1/recall)) ;

    cout << "labeled= "<<count_1<<" predicted="<<count_2<<" True="<<count_true<<endl;
    cout << "Precision= "<<precision<<endl;
    cout << "Recall=:"<<recall<<endl;
    cout << "F-score= "<< f_score << endl;

    return 0;

}




