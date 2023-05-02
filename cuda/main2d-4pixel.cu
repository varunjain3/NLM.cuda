#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper2d.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// #def data folder
// #define sWindowSize 21
// #define nWindowSize 13
#define sWindowSize 23
#define nWindowSize 11
#define H .5
#define image_path "../sp_noise/Image3.png"
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024
#define MAX_THREAD_DIM 16

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

    int i = threadIdx.x;
    int j = threadIdx.y;

    int padI = blockIdx.x * blockDim.x + threadIdx.x;
    int padJ = blockIdx.y * blockDim.y + threadIdx.y;

    // using threads to bring data to the shared memory
    for (int it = 0; it < MAX_THREAD_DIM_SHARED; it += MAX_THREAD_DIM)
    {
        for (int jt = 0; jt < MAX_THREAD_DIM_SHARED; jt++)
        {
            share_buf[i + it][jt] = paddedImage[(blockIdx.x * blockDim.x + i + it) * width + (blockIdx.y * blockDim.y + jt)];
        }
    }
    __syncthreads();

    float weightedSum = 0;
    float similaritySum = 0;
    float h_const = 1 / (h);

#pragma unroll
    for (int k = 0; k < sWindowSize; k++)
    {
#pragma unroll
        for (int l = 0; l < sWindowSize; l++)
        {
            float dist = 0;
#pragma unroll
            for (int m = 0; m < nWindowSize; m++)
            {
#pragma unroll
                for (int n = 0; n < nWindowSize; n++)
                {
                    float temp_dist;
                    temp_dist = share_buf[k + m + i][l + n + j] - share_buf[m + i + halfSearchWindowSize][n + j + halfSearchWindowSize];
                    dist += temp_dist * temp_dist;
                }
                // float dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9, dist10, dist11;
                // dist1 = share_buf[k + m + i][l + 0 + j] - share_buf[m + i + halfSearchWindowSize][0 + j + halfSearchWindowSize];
                // dist2 = share_buf[k + m + i][l + 1 + j] - share_buf[m + i + halfSearchWindowSize][1 + j + halfSearchWindowSize];
                // dist3 = share_buf[k + m + i][l + 2 + j] - share_buf[m + i + halfSearchWindowSize][2 + j + halfSearchWindowSize];
                // dist4 = share_buf[k + m + i][l + 3 + j] - share_buf[m + i + halfSearchWindowSize][3 + j + halfSearchWindowSize];
                // dist5 = share_buf[k + m + i][l + 4 + j] - share_buf[m + i + halfSearchWindowSize][4 + j + halfSearchWindowSize];
                // dist6 = share_buf[k + m + i][l + 5 + j] - share_buf[m + i + halfSearchWindowSize][5 + j + halfSearchWindowSize];
                // dist7 = share_buf[k + m + i][l + 6 + j] - share_buf[m + i + halfSearchWindowSize][6 + j + halfSearchWindowSize];
                // dist8 = share_buf[k + m + i][l + 7 + j] - share_buf[m + i + halfSearchWindowSize][7 + j + halfSearchWindowSize];
                // dist9 = share_buf[k + m + i][l + 8 + j] - share_buf[m + i + halfSearchWindowSize][8 + j + halfSearchWindowSize];
                // dist10 = share_buf[k + m + i][l + 9 + j] - share_buf[m + i + halfSearchWindowSize][9 + j + halfSearchWindowSize];
                // dist11 = share_buf[k + m + i][l + 10 + j] - share_buf[m + i + halfSearchWindowSize][10 + j + halfSearchWindowSize];
                // dist1 = dist1 * dist1;
                // dist2 = dist2 * dist2;
                // dist3 = dist3 * dist3;
                // dist4 = dist4 * dist4;
                // dist5 = dist5 * dist5;
                // dist6 = dist6 * dist6;
                // dist7 = dist7 * dist7;
                // dist8 = dist8 * dist8;
                // dist9 = dist9 * dist9;
                // dist10 = dist10 * dist10;
                // dist11 = dist11 * dist11;
                // dist += dist1 + dist2 + dist3 + dist4 + dist5 + dist6 + dist7 + dist8 + dist9 + dist10 + dist11;
            }
            dist = sqrt(dist);
            float w = exp(-dist * h_const);

            weightedSum += w * share_buf[k + i + halfWindowSize][l + j + halfWindowSize];
            similaritySum += w;
        }
    }
    float intensity = weightedSum / similaritySum;

    outputImage[padI * cols + padJ] = intensity;
    __syncthreads();
}

int findSqaureNum(int n)
{
    // find square number less than or equal to n
    int i = 1;
    while (i * i <= n)
        i++;
    return i - 1;
}

cv::Mat NL_Means(cv::Mat src, int MAX_BLOCK_DIM)
{
    int rows = src.rows;
    int cols = src.cols;

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
    const int imgSize = stoi(argv[1]);
    const int MAX_BLOCK_DIM = imgSize / MAX_THREAD_DIM;
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    src = resizeImage(src, imgSize, imgSize);
    cout << "Shape of image: " << src.size() << endl;

    // checkDevice();

    if (src.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    auto start_time = std::chrono::steady_clock::now();
    NL_Means(src, MAX_BLOCK_DIM);
    auto end_time = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Runtime: " << runtime.count() << "ms" << std::endl;

    return 0;
}