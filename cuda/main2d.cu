#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper2d.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// #def data folder
#define imgH 256
#define imgW 256
// #define sWindowSize 21
// #define nWindowSize 13
#define sWindowSize 23
#define nWindowSize 11
#define H .5
#define image_path "../sp_noise/Image3.png"
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024
#define MAX_THREAD_DIM 32
#define MAX_BLOCK_DIM 256/32

// run using: g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`
// run using: nvcc -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`

__global__ void perPixel(float *paddedImage, float *outputImage, int width, int height, int rows, int cols)
{

    float h = H;
    const int halfWindowSize = nWindowSize / 2;
    const int halfSearchWindowSize = sWindowSize / 2;

    const int MAX_THREAD_DIM_SHARED = MAX_THREAD_DIM + 2 * halfWindowSize + 2 * halfSearchWindowSize;
    // 2d shared memory according to block size
    __shared__ float share_buf[MAX_THREAD_DIM_SHARED][MAX_THREAD_DIM_SHARED];

    // calculate i and j with blocksize and threadid
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = threadIdx.x;
    int j = threadIdx.y;

    // int padI = 1 * MAX_THREAD_DIM + threadIdx.x;
    // int padJ = 1 * MAX_THREAD_DIM + threadIdx.y;
    int padI = blockIdx.x * blockDim.x + threadIdx.x;
    int padJ = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("threadIdx.x: %d ; threadIdx.y: %d ", threadIdx.x, threadIdx.y);

    // if (threadIdx.x == 0 and threadIdx.y == 0 and blockIdx.x == 0 and blockIdx.y == 0)
    // {
    //     printf("MAX_THREAD_DIM_SHARED: %d \n", MAX_THREAD_DIM_SHARED);
    //     printf("i: %d ; j: %d \n", i, j);
    //     printf("blockDim.x: %d ; blockDim.y: %d \n", blockDim.x, blockDim.y);
    //     printf("blockIdx.x: %d ; blockIdx.y: %d \n", blockIdx.x, blockIdx.y);
    //     printf("threadIdx.x: %d ; threadIdx.y: %d \n", threadIdx.x, threadIdx.y);
    //     printf("height: %d ; width: %d \n", height, width);
    //     printf("rows: %d ; cols: %d \n", rows, cols);
    // }

    // int padI = halfSearchWindowSize + halfWindowSize + i;
    // int padJ = halfSearchWindowSize + halfWindowSize + j;

    // let 1 thread load all the data into shared memory
    // TODO: This can technically be distributed to all threads smartly
    // printf("blockIdx.x: %d ; blockIdx.y: %d; i: %d ; j: %d threadId.x: %d ; threadId.y: %d \n", blockIdx.x, blockIdx.y, i, j, threadIdx.x, threadIdx.y);

    if (threadIdx.x == 0 and threadIdx.y == 0)
    {

        for (int it = 0; it < MAX_THREAD_DIM_SHARED; it++)
        {
            for (int jt = 0; jt < MAX_THREAD_DIM_SHARED; jt++)
            {
                // printf("i: %d ; j: %d \nit: %d ; jt: %d \n\n", i, j, it, jt);
                share_buf[it][jt] = paddedImage[(padI + it) * width + (padJ + jt)];
            }
        }
    }

    // printf("threadIdx.x: %d ; threadIdx.y: %d ", threadIdx.x, threadIdx.y);

    __syncthreads();

    float weightedSum = 0;
    float similaritySum = 0;

    for (int k = 0; k < sWindowSize; k++)
    {
        for (int l = 0; l < sWindowSize; l++)
        {
            float dist = 0;
            for (int m = 0; m < nWindowSize; m++)
            {
                for (int n = 0; n < nWindowSize; n++)
                {
                    // if (threadIdx.x == 0 and threadIdx.y == 0)
                    // {
                    //     printf("k + m + i: %d ; l + n + j: %d \nm + i + halfSearchWindowSize: %d n + j +halfSearchWindowSize: %d\n", k + m + i, l + n + j, m + i + halfSearchWindowSize, n + j + halfSearchWindowSize);
                    // }
                    dist += pow(share_buf[k + m + i][l + n + j] - share_buf[m + i + halfSearchWindowSize][n + j + halfSearchWindowSize], 2);
                }
            }
            // cout<<dist<<endl;
            dist = sqrt(dist);
            // // cout<<dist<<endl;
            float w = exp(-dist / (h));

            // if (threadIdx.x == 0 and threadIdx.y == 0)
            // {
            //     printf("k: %d ; l: %d i: %d ; j: %d \n", k, l, i, j);
            //     printf("k + i halfWindowSize: %d ; k + i halfWindowSize: %d \n\n", k + i + halfWindowSize, l + j + halfWindowSize);
            // }
            weightedSum += w * share_buf[k + i + halfWindowSize][l + j + halfWindowSize];
            similaritySum += w;
        }
    }
    float intensity = weightedSum / similaritySum;

    // printf("i: %d ; j: %d; i * cols + j: %d \n", i, j, i * cols + j);
    outputImage[padI * cols + padJ] = intensity;
    __syncthreads();
}

// __syncthreads();
// for (int it = 0; it < blockDim.x; it++)
// {
//     for (int jt = 0; jt < blockDim.y; jt++)
//     {
//         outputImage[i * rows + jt] = share_buf[it][jt];
//     }
// }

// for (int jt = 0; jt < cols; jt++)
// {
//     // printf("here");
//     outputImage[i * rows + jt] = share_buf[jt];
// }

int findSqaureNum(int n)
{
    // find square number less than or equal to n
    int i = 1;
    while (i * i <= n)
        i++;
    return i - 1;
}

cv::Mat NL_Means(cv::Mat src)
{
    int rows = src.rows;
    int cols = src.cols;

    int h = H;
    int halfWindowSize = nWindowSize / 2;
    int halfSearchWindowSize = sWindowSize / 2;
    // cout << "nWindowSize: " << nWindowSize << endl;
    // cout << "sWindowSize: " << sWindowSize << endl;

    // cout << "Performing NL_Means on the Image" << endl;

    vector<vector<float>> paddedImageVec = padImage(src, sWindowSize);
    paddedImageVec = floatImage(paddedImageVec);

    float *paddedImage = vector2float(paddedImageVec);
    float *outputImage = (float *)malloc(rows * cols * sizeof(float));

    vector<int> sizes = get_sizes(paddedImageVec);
    int height = sizes[0];
    int width = sizes[1];

    float *devicePaddedImage;
    float *deviceOutputImage;

    cudaError_t err = cudaMalloc(&devicePaddedImage, height * width * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }
    cudaError_t err2 = cudaMalloc(&deviceOutputImage, rows * cols * sizeof(float));
    if (err2 != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // cout << "main rows: " << rows << " ; cols: " << cols << endl;
    // cout << "padd rows: " << height << " ; cols: " << width << endl;

    cudaMemcpy(devicePaddedImage, paddedImage, height * width * sizeof(float), cudaMemcpyHostToDevice);

    //  define 2d block and grid size
    // int threadDim = findSqaureNum(MAX_THREADS);
    dim3 blockSize(MAX_THREAD_DIM, MAX_THREAD_DIM);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);
    dim3 check(MAX_BLOCK_DIM, MAX_BLOCK_DIM);

    // cout << "Block size: " << blockSize.x << "x" << blockSize.y << endl;
    // cout << "Grid size: " << gridSize.x << "x" << gridSize.y << endl;

    perPixel<<<check, blockSize>>>(devicePaddedImage, deviceOutputImage, width, height, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage, deviceOutputImage, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    vector<vector<float>> new_outputImage = intImage(outputImage, rows, cols);

    // cout << "Saving Image" << endl;

    cv::Mat dst;
    dst = Vec2Mat(new_outputImage, "outputImage.png");

    cudaFree(devicePaddedImage);
    cudaFree(deviceOutputImage);
    free(paddedImage);
    free(outputImage);

    return dst;
}

// check device properties
// void checkDevice()
// {
//     int device = 0;
//     int max_blocks_per_multiprocessor;
//     int num_multiprocessors;
//     int max_blocks_total;
//     cudaDeviceProp prop;

//     cudaGetDeviceProperties(&prop, device);
//     cudaDeviceGetAttribute(&max_blocks_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device);
//     cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, device);
//     cudaDeviceGetAttribute(&max_blocks_total, cudaDevAttrMaxBlockDimX, device);

//     int max_blocks = min(max_blocks_per_multiprocessor * num_multiprocessors, max_blocks_total);

//     printf("\n---------------Device Properties---------------\n");
//     printf("Device name: %s\n", prop.name);
//     printf("Compute capability: %d.%d\n", prop.major, prop.minor);
//     printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
//     printf("Maximum number of blocks: %d\n", max_blocks);
//     printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
//     printf("Amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
//     printf("-----------------------------------------------\n\n");
// }

int main(int argc, char **argv)
{
    auto start_time = std::chrono::steady_clock::now(); 
    // string image_path
    // cout << "Loading image " << image_path << endl;
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    src = resizeImage(src, imgH, imgW);
    // cout << "Shape of image: " << src.size() << endl;

    // checkDevice();

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    NL_Means(src);

    // cout << "Done!" << endl;
    auto end_time = std::chrono::steady_clock::now(); 
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time); 
    
    std::cout << "Runtime: " << runtime.count() << "ms" << std::endl; 

    return 0;
}