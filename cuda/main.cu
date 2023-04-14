#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper.h"

#define imgH 1024
#define imgW 1024
#define sWindowSize 7
#define nWindowSize 3
#define image_path "../sp_noise/Image3.png"

using namespace std;

// #def data folder

// run using: g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`
// run using: nvcc -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`

// /*
__global__ void pixel_kernel_call(float* paddedImage, float* outputImage, int cols, int windowSize, int searchWindowSize, int h, int i) {
    // printf("Hello World!");
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("j: %d", j);
    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    float weightedSum = 0;
    float similaritySum = 0;

    for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
        for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
            float dist = 0;
            for(int m=-halfWindowSize; m<=halfWindowSize; m++){
                for(int n=-halfWindowSize; n<=halfWindowSize; n++){
                    dist += pow(paddedImage[(i+k+halfSearchWindowSize)*cols + (j+l+halfSearchWindowSize)] - paddedImage[(i+m+halfSearchWindowSize)*cols + (j+n+halfSearchWindowSize)], 2);
                }
            }
            // cout<<dist<<endl;
            dist = sqrt(dist);
            // cout<<dist<<endl;
            float w = exp(-dist/(h));

            weightedSum += w*paddedImage[(i+k+halfSearchWindowSize)*cols + (j+l+halfSearchWindowSize)];
            similaritySum += w;
        }
    }
    float intensity = weightedSum/similaritySum;
    // cout<<intensity<<endl;
    outputImage[i*cols + j] = intensity;
}

// */

cv::Mat NL_Means(cv::Mat src, int h = 2, int windowSize = 3, int searchWindowSize = 7)
{
    int rows = src.rows;
    int cols = src.cols;

    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    cout << "Performing NL_Means on the Image" << endl;

    vector<vector<float>> paddedImage_temp = padImage(src, searchWindowSize);

    paddedImage_temp = floatImage(paddedImage_temp);
    float* paddedImage = vec_to_float_arr(paddedImage_temp);

    float *outputImage;
    float *dev_pad, *dev_out_img;

    // vector<vector<float>> outputImage

    // size_t N = 128;
    vector<int> sizes = get_sizes(paddedImage_temp);
    size_t pad_arr_len = sizes[0] * sizes[1];
    outputImage = (float *)malloc(pad_arr_len*sizeof(float));
    // size_t pad_arr_len = paddedImage.size() * paddedImage[0].size();
    
    //create buffer on device
    cudaError_t err = cudaMalloc(&dev_pad, pad_arr_len*sizeof(float));
    if (err != cudaSuccess){
        cout<<"Dev Memory not allocated"<<endl;
        exit(-1);
    }
    cudaError_t err2 = cudaMalloc(&dev_out_img, pad_arr_len*sizeof(float));
    if (err2 != cudaSuccess){
        cout<<"Dev Memory not allocated"<<endl;
        exit(-1);
    }

    cudaMemcpy(dev_pad, paddedImage, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);
    size_t threads = 16;
    dim3 threadsperblock(threads, threads);
    dim3 numBlocks(threads*threads/threadsperblock.x,  threads*threads/threadsperblock.y);

    // print parameters
    cout << "Threads: " << threads << endl;
    cout << "Blocks: " << numBlocks.x << "x" << numBlocks.y << endl;
    cout << "Threads per block: " << threadsperblock.x << "x" << threadsperblock.y << endl;

    //CREATE MULTIPLE STREAMS HERE
    int num_streams = 4;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
      cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    cout<<"main rows: "<<rows<<" ; cols: "<< cols<< endl;
    for (int i = 0; i < rows; i++)
    {
    //     cout << i << endl;
    //     for (int j = 0; j < cols; j++)
    //     {
    pixel_kernel_call<<<1, 128, 0, streams[0]>>>(dev_pad, dev_out_img, cols, windowSize, searchWindowSize, h, i);
    //     }
    }

    cudaMemcpy(outputImage, dev_out_img, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Done" << endl;

    vector<vector<float> > new_outputImage = intImage(outputImage, sizes[0], sizes[1]);
    cout<<"Saving image"<< endl;
    cv::Mat dst = Vec2Mat(new_outputImage, "outputImage.png");

    return dst;
}

int main(int argc, char **argv)
{
    int searchWindowSize = sWindowSize;
    int windowSize = nWindowSize;

    // string image_path
    cout << "Loading image " << image_path << endl;

    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    // save original shape of image
    int OrignalH = src.rows;
    int OrignalW = src.cols;

    cv::resize(src, src, cv::Size(imgH - 2 * searchWindowSize, imgW - 2 * searchWindowSize));


    

    cout << "Shape of image: " << src.size() << endl;

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    
    cv::Mat dst = NL_Means(src, 2, windowSize, searchWindowSize);

    cv::resize(dst, dst, cv::Size(OrignalH, OrignalW));
    cv::imwrite("output.png", dst);

    return 0;
}