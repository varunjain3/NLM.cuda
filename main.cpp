#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper.h"

using namespace std;

// #def data folder
#define image_path "sp_noise/Image3.png"

// run using: g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`

cv::Mat NL_Means(cv::Mat src, int h = 2, int windowSize=3, int searchWindowSize=7)
{
    int rows = src.rows;
    int cols = src.cols;

    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;


    cout << "Performing NL_Means on the Image" << endl;

    vector<vector<float> > paddedImage = padImage(src, searchWindowSize);
    paddedImage = floatImage(paddedImage);

    vector<vector<float> > outputImage(rows, vector<float>(cols));

    for(int i=0; i<rows;i++){
        cout<<i<<endl;
        for(int j=0; j<cols; j++){
            float weightedSum = 0;
            float similaritySum = 0;

            for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
                for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
                    float dist = 0;
                    float tempWeightedSum = 0;
                    float tempSimilaritySum = 0;
                    for(int m=-halfWindowSize; m<=halfWindowSize; m++){
                        for(int n=-halfWindowSize; n<=halfWindowSize; n++){
                            dist += pow(paddedImage[i+k+halfSearchWindowSize][j+l+halfSearchWindowSize] - paddedImage[i+m+halfSearchWindowSize][j+n+halfSearchWindowSize], 2);
                            // cout<< i+k+halfSearchWindowSize << " : "<< j+l+halfSearchWindowSize << " \n ";
                            // cout<< i+m+halfSearchWindowSize << " : "<< j+n+halfSearchWindowSize <<endl;
                            // if (i+k+halfSearchWindowSize < 0 || j+l+halfSearchWindowSize < 0 || i+m+halfSearchWindowSize < 0 || j+n+halfSearchWindowSize < 0){
                            //     cout<<"negative index"<<endl;
                            // }
                        }
                    }
                    // cin>>dist;
                    // cout<<dist<<endl;
                    dist = sqrt(dist);
                    // cout<<dist<<endl;
                    float w = exp(-dist/(h));

                    weightedSum += w*paddedImage[i+k+halfSearchWindowSize][j+l+halfSearchWindowSize];
                    similaritySum += w;
                }
            }
            float intensity = weightedSum/similaritySum;
            // cout<<intensity<<endl;
            outputImage[i][j] = intensity;

        }
    }

    outputImage = intImage(outputImage);

    // for (int i = 0; i < outputImage.size(); i++)
    // {
    //     for (int j = 0; j < outputImage[0].size(); j++)
    //     {
    //         cout<<outputImage[i][j]<<" ";
    //     }
    //     break;
    // }

    cv::Mat dst = Vec2Mat(outputImage, "outputImage.png");
    
    return dst;
}

int main(int argc, char **argv)
{

    // string image_path
    cout << "Loading image " << image_path << endl;

    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cout << "Shape of image: " << src.size() << endl;

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    NL_Means(src);

    return 0;
}