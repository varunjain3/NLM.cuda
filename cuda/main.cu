#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper.h"

using namespace std;

// #def data folder
#define image_path "../sp_noise/Image3.png"

// run using: g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`


void check_func(float* outputImage, int cols, int halfWindowSize, int halfSearchWindowSize, int h, int rows, float* paddedImage1){
    // float** paddedImage = new float*[270];
    // for (int i = 0; i < 270; i++)
    //     paddedImage[i] = new float[270];
    
    // for (int i = 0; i < 270; i++){
    //     for (int j = 0; j < 270; j++){
    //         paddedImage[i][j] = paddedImage1[i*270 + j];
    //     }
    // }

    for(int i=0; i<rows;i++){
        // cout<<i<<endl;
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
                            dist += pow(paddedImage1[(i+k+halfSearchWindowSize)*270 + (j+l+halfSearchWindowSize)] - 
                            paddedImage1[(i+m+halfSearchWindowSize)*270 + (j+n+halfSearchWindowSize)], 2);
                        }
                    }
                    // cout<<dist<<endl;
                    dist = sqrt(dist);
                    // cout<<dist<<endl;
                    float w = exp(-dist/(h));

                    weightedSum += w*paddedImage1[(i+k+halfSearchWindowSize)*270 + (j+l+halfSearchWindowSize)];
                    similaritySum += w;
                }
            }
            float intensity = weightedSum/similaritySum;
            // cout<<intensity<<endl;
            outputImage[i*rows + j] = intensity;

        }
    }
}


__global__ void pixel_kernel_call(float* paddedImage, float* outputImage, int cols, int halfWindowSize, int halfSearchWindowSize, int h, int rows, int i){
    // printf("cols: %d", cols);
    __shared__ float share_buf[256];

    //  printf("i: %d", i);
    for (int xx = 0;xx < cols/128; xx++) {
        int j = threadIdx.x + (128*xx);
        // printf("j: %d\n", j);
        if (j >= cols) {
            printf("Thread exceeded cols: %d", j);
            break;
        }

        float weightedSum = 0;
        float similaritySum = 0;

        for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
            for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
                float dist = 0;
                float tempWeightedSum = 0;
                float tempSimilaritySum = 0;
                for(int m=-halfWindowSize; m<=halfWindowSize; m++){
                    for(int n=-halfWindowSize; n<=halfWindowSize; n++){
                        dist += pow(paddedImage[(i+k+halfSearchWindowSize)*270 + (j+l+halfSearchWindowSize)] - 
                        paddedImage[(i+m+halfSearchWindowSize)*270 + (j+n+halfSearchWindowSize)], 2);
                    }
                }
                // cout<<dist<<endl;
                dist = sqrt(dist);
                // cout<<dist<<endl;
                float w = exp(-dist/(h));

                weightedSum += w*paddedImage[(i+k+halfSearchWindowSize)*270 + (j+l+halfSearchWindowSize)];
                similaritySum += w;
            }
        }
        float intensity = weightedSum/similaritySum;
        share_buf[j] = intensity;
        // cout<<intensity<<endl;
        // outputImage[i][j] = intensity;
    }

    __syncthreads();
        for (int jt = 0;jt < cols; jt++) {
            // printf("here");
            outputImage[i*rows + jt] = share_buf[jt];
        }
    
}


cv::Mat NL_Means(cv::Mat src, int h = 2, int windowSize=3, int searchWindowSize=7)
{
    int rows = src.rows;
    int cols = src.cols;

    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;


    cout << "Performing NL_Means on the Image" << endl;

   
    vector<vector<float>> paddedImage_temp = padImage(src, searchWindowSize);
    paddedImage_temp = floatImage(paddedImage_temp);

    float* paddedImage = vec_to_float_arr(paddedImage_temp, 0);

   
    float* outputImage = (float*) malloc(rows * cols * sizeof(float));
    
    vector<int> sizes = get_sizes(paddedImage_temp);
    size_t pad_arr_len = sizes[0] * sizes[1];
    
    float* dev_pad;
    float* dev_out_img;
     //create buffer on device
    cudaError_t err = cudaMalloc(&dev_pad, pad_arr_len*sizeof(float));
    if (err != cudaSuccess){
        cout<<"Dev Memory not allocated"<<endl;
        exit(-1);
    }
    cudaError_t err2 = cudaMalloc(&dev_out_img, rows*cols*sizeof(float));
    if (err2 != cudaSuccess){
        cout<<"Dev Memory not allocated"<<endl;
        exit(-1);
    }

   
    cout<<"main rows: "<<rows<<" ; cols: "<< cols<< endl;
    cout<<"padd rows: "<<sizes[0]<<" ; cols: "<< sizes[1]<< endl;
    cudaMemcpy(dev_pad, paddedImage, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);
    //CREATE MULTIPLE STREAMS HERE
    int num_streams = 16;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    // for (int i = 0; i < rows; i++)
    // {
    // cout<<"Starting kernel call"<<endl;
    for (int i = 0; i < rows;i++){
        pixel_kernel_call<<<1, 128, 0, streams[i%num_streams]>>>(dev_pad, dev_out_img, cols, halfWindowSize, halfSearchWindowSize, h, rows, i);
    }
   
    cudaDeviceSynchronize();
    cudaMemcpy(outputImage, dev_out_img, rows*cols * sizeof(float), cudaMemcpyDeviceToHost);
   
    vector<vector<float> > new_outputImage = intImage(outputImage, rows, cols);

    cout<<"Saving Image"<<endl;

    cv::Mat dst = Vec2Mat(new_outputImage, "outputImage.png");
    
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