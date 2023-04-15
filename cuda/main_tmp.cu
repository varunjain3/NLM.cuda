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

__global__ void pixel_kernel_call(float* paddedImage, float* outputImage, int cols, int windowSize, int searchWindowSize, int h, int rows, int row_index) {
    // printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

     // cout << i << endl;
    __shared__ float share_buf[1010];

    for (int xx = 0;xx <= cols/128; xx++) {
        // printf("xx: %d", xx);
        int j = threadIdx.x + (128*xx);
        // printf("j: %d\n", j);
        if (j < cols) {

            float weightedSum = 0;
            float similaritySum = 0;

            for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
                for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
                    float dist = 0;
                    for(int m=-halfWindowSize; m<=halfWindowSize; m++){
                        for(int n=-halfWindowSize; n<=halfWindowSize; n++){
                            dist += pow( 
                                paddedImage[(row_index+k+halfSearchWindowSize)*rows + (j+l+halfSearchWindowSize)] - 
                                paddedImage[(row_index+m+halfSearchWindowSize)*rows + (j+n+halfSearchWindowSize)], 2);
                            // dist = dist + (data*data);
                        }
                    }
                    // cout<<dist<<endl;
                    dist = sqrt(dist);
                    // cout<<dist<<endl;
                    float w = exp(-dist/(h));

                    weightedSum += w*paddedImage[(row_index+k+halfSearchWindowSize)*rows + (j+l+halfSearchWindowSize)];
                    similaritySum += w;
                }
            }
            float intensity = weightedSum/similaritySum;
            // cout<<intensity<<endl;
            share_buf[j] = intensity;
        }
    }
    __syncthreads();
    for (int jt = 0; jt < cols; jt++)
    {
        outputImage[row_index*rows + jt] = share_buf[jt];
    }

    
}

void check_func(float** paddedImage, float* outputImage, int cols, int windowSize, int searchWindowSize, int h, int rows) {
    // printf("%d %d %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    for (int row_index = 0; row_index < rows; row_index++)
    {
        // cout << i << endl;
        float share_buf[1010];

        for (int j = 0;j < cols; j++) {

            float weightedSum = 0;
            float similaritySum = 0;

            for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
                for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
                    float dist = 0;
                    for(int m=-halfWindowSize; m<=halfWindowSize; m++){
                        for(int n=-halfWindowSize; n<=halfWindowSize; n++){
                            dist += pow( 
                                paddedImage[(row_index+k+halfSearchWindowSize)][ (j+l+halfSearchWindowSize)] - 
                                paddedImage[(row_index+m+halfSearchWindowSize)][(j+n+halfSearchWindowSize)], 2);
                            // dist = dist + (data*data);
                        }
                    }
                    // cout<<dist<<endl;
                    dist = sqrt(dist);
                    // cout<<dist<<endl;
                    float w = exp(-dist/(h));

                    weightedSum += w*paddedImage[(row_index+k+halfSearchWindowSize)][(j+l+halfSearchWindowSize)];
                    similaritySum += w;
                }
            }
            float intensity = weightedSum/similaritySum;
            // cout<<intensity<<endl;
            share_buf[j] = intensity;
        
        }
        for (int jt = 0; jt < cols; jt++)
        {
            outputImage[row_index*rows + jt] = share_buf[jt];
        }
    }

}

cv::Mat NL_Means(cv::Mat src, int h = 2, int windowSize = 3, int searchWindowSize = 7)
{
    int rows = src.rows;
    int cols = src.cols;

    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    cout << "Performing NL_Means on the Image" << endl;

    vector<vector<float>> paddedImage_temp = padImage(src, searchWindowSize);
    paddedImage_temp = floatImage(paddedImage_temp);

    float** paddedImage = vec_to_float_arr(paddedImage_temp);
    // saving this pad image to check for correctness
    // Vec2Mat(paddedImage, 1024, 1024, "paddedImage_use.png");

     vector<int> sizes = get_sizes(paddedImage_temp);
    size_t pad_arr_len = sizes[0] * sizes[1];


    float *outputImage;
    float *dev_pad, *dev_out_img;

    outputImage = (float *)malloc(rows*cols*sizeof(float));
    // size_t pad_arr_len = paddedImage.size() * paddedImage[0].size();
    
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

    cudaMemcpy(dev_pad, paddedImage, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);
    // size_t threads = 16;
    // dim3 threadsperblock(threads, threads);
    // dim3 numBlocks(threads*threads/threadsperblock.x,  threads*threads/threadsperblock.y);

    // // print parameters
    // cout << "Threads: " << threads << endl;
    // cout << "Blocks: " << numBlocks.x << "x" << numBlocks.y << endl;
    // cout << "Threads per block: " << threadsperblock.x << "x" << threadsperblock.y << endl;

    //CREATE MULTIPLE STREAMS HERE
    // int num_streams = 16;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++) {
    //   cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    // }

    // cout<<"main rows: "<<rows<<" ; cols: "<< cols<< endl;

    // for (int i = 0; i < rows; i++)
    // {
    //    pixel_kernel_call<<<1, 128, 0, streams[i%num_streams]>>>(dev_pad, dev_out_img, cols, windowSize, searchWindowSize, h, rows, i);
    //     // pixel_kernel_call<<<1, 128>>>(dev_pad, dev_out_img, cols, windowSize, searchWindowSize, h, rows, i);

    // }
    // cudaDeviceSynchronize();

    // cudaMemcpy(outputImage, dev_out_img, rows*cols * sizeof(float), cudaMemcpyHostToDevice);
    
        check_func(paddedImage, outputImage, cols, windowSize, searchWindowSize, h, rows);



    cout << "Done" << endl;

    vector<vector<float> > new_outputImage = intImage(outputImage, rows, cols);
    cout<<"Saving image"<< endl;
    

    cv::Mat dst = Vec2Mat(new_outputImage, "outputImage.png");

    free(paddedImage);
    free(outputImage);
    cudaFree(dev_pad);
    cudaFree(dev_out_img);

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