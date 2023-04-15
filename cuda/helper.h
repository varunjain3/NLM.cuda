#include <opencv2/opencv.hpp>
#include <iostream>

#define imgH 1024
#define imgW 1024

using namespace std;

cv::Mat Vec2Mat(vector<vector<float> > image, string name = "temp.png"){
    int rows = image.size();
    int cols = image[0].size();
    
    // one channel image
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst.at<double>(i, j) = image[i][j];

    cv::imwrite(name, dst);

    return dst;
}

void Vec2Mat(float* image, int rows, int cols, string name = "temp.png"){
    // int rows = image.size();
    // int cols = image[0].size();
    
    // one channel image
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst.at<double>(i, j) = image[i*rows+j];

    cv::imwrite(name, dst);

    
    // return dst;
}

vector<vector<float> > padImage(cv::Mat image, int padding)
{

    int rows = image.rows;
    int cols = image.cols;

    int newRows = rows + 2 * padding;
    int newCols = cols + 2 * padding;

    vector<vector<float> > paddedImage(newRows, vector<float>(newCols));

    for (int i = 0; i < newRows; i++)
    {
        for (int j = 0; j < newCols; j++)
        {
            int img_i = i - padding;
            int img_j = j - padding;
            if (i < padding)
                img_i = padding - i;
            if (j < padding)
                img_j = padding - j;
            if (i >= rows + padding)
                img_i -= 2 * (i - padding - rows) + 1;
            if (j >= cols + padding)
                img_j -= 2 * (j - padding - cols) + 1;

            paddedImage[i][j] = (float)image.at<uchar>(img_i, img_j);
        }
    }

    Vec2Mat(paddedImage,"paddedImage.png");

    return paddedImage;
}

vector<vector<float> > floatImage(vector<vector<float> > image)
{
    int rows = image.size();
    int cols = image[0].size();

    vector<vector<float> > floatImage(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            floatImage[i][j] = (float)(image[i][j]/255.0);
        }
    }

    return floatImage;
}

vector<vector<float> > intImage(vector<vector<float> > image){
    int rows = image.size();
    int cols = image[0].size();

    vector<vector<float> > intImage(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            intImage[i][j] = (int)(image[i][j]*255.0);
        }
    }

    return intImage;
}


vector<vector<float> > intImage(float** image, int rows, int cols){
    vector<vector<float> > intImage(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            intImage[i][j] = (int)(image[i][j]*255.0);
        }
    }

    return intImage;
}


vector<vector<float> > intImage(float* image, int rows, int cols){
    vector<vector<float> > intImage(rows, vector<float>(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            intImage[i][j] = (int)(image[i*rows + j]*255.0);
        }
    }

    return intImage;
}

float** vec_to_float_arr(vector<vector<float> > image){
    int rows = image.size();
    int cols = image[0].size();

    float** floatImage = new float*[rows];


    for (int i = 0; i < rows; i++)
    {
        floatImage[i] = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            floatImage[i][j] = image[i][j];
        }
    }

    return floatImage;
}

float* vec_to_float_arr(vector<vector<float> > image, int data){
    int rows = image.size();
    int cols = image[0].size();

    float* floatImage = new float[rows*cols];


    for (int i = 0; i < rows; i++)
    {
        // floatImage[i] = new float[cols];
        for (int j = 0; j < cols; j++)
        {
            floatImage[i*rows + j] = image[i][j];
        }
    }

    return floatImage;
}

vector<int> get_sizes(vector<vector<float> > image){
    vector<int> sizes;
    sizes.push_back(image.size());
    sizes.push_back(image[0].size());
    return sizes;
}