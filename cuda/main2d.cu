#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "helper2d.h"
#include <cuda_runtime.h>

using namespace std;

// #def data folder
#define imgH 1024
#define imgW 1024
#define sWindowSize 7
#define nWindowSize 3
#define H 2
#define image_path "../sp_noise/Image3.png"
#define MAX_THREADS 1024
#define MAX_BLOCKS 1024

// run using: g++ -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`
// run using: nvcc -std=c++11 main.cpp -o main `pkg-config --cflags --libs opencv`

__global__ void pixel_kernel_call(float *paddedImage, float *outputImage)
{
    // 2d shared memory according to block size
    __shared__ float share_buf[blockDim.x][blockDim.y];

    int h = H;
    int halfWindowSize = nWindowSize / 2;
    int halfSearchWindowSize = sWindowSize / 2;

    // calculate i and j with blocksize and threadid
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (blockId.x == 0 and blockId.y == 0)
    {
        printf("i: %d", i);
        printf("j: %d", j);
    }

    float weightedSum = 0;
    float similaritySum = 0;

    for (int k = -halfSearchWindowSize; k <= halfSearchWindowSize; k++)
    {
        for (int l = -halfSearchWindowSize; l <= halfSearchWindowSize; l++)
        {
            float dist = 0;
            for (int m = -halfWindowSize; m <= halfWindowSize; m++)
            {
                for (int n = -halfWindowSize; n <= halfWindowSize; n++)
                {
                    dist += pow(paddedImage[(i + k + halfSearchWindowSize) * 270 + (j + l + halfSearchWindowSize)] -
                                    paddedImage[(i + m + halfSearchWindowSize) * 270 + (j + n + halfSearchWindowSize)],
                                2);
                }
            }
            // cout<<dist<<endl;
            dist = sqrt(dist);
            // cout<<dist<<endl;
            float w = exp(-dist / (h));

            weightedSum += w * paddedImage[(i + k + halfSearchWindowSize) * 270 + (j + l + halfSearchWindowSize)];
            similaritySum += w;
        }
    }
    float intensity = weightedSum / similaritySum;
    share_buf[j] = intensity;
    // cout<<intensity<<endl;
    // outputImage[i][j] = intensity;
}

__syncthreads();
for (int it = 0; it < blockDim.x; it++)
{
    for (int jt = 0; jt < blockDim.y; jt++)
    {
        outputImage[i * rows + jt] = share_buf[it][jt];
    }
}

for (int jt = 0; jt < cols; jt++)
{
    // printf("here");
    outputImage[i * rows + jt] = share_buf[jt];
}

int findSqaureNum(int n)
{
    // find square number less than or equal to n
    int i = 1;
    while (i * i <= n)
        i++;
    return i - 1;
}

cv::Mat resizeImage(cv::Mat src)
{
    // resize to imgH x imgW
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(imgW, imgH), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("resizedImage.png", dst);
    return dst;
}

cv::Mat NL_Means(cv::Mat src)
{
    int rows = src.rows;
    int cols = src.cols;

    int h = H;
    int halfWindowSize = nWindowSize / 2;
    int halfSearchWindowSize = sWindowSize / 2;

    cout << "Performing NL_Means on the Image" << endl;

    vector<vector<float>> paddedImageVec = padImage(src, searchWindowSize);
    paddedImageVec = floatImage(paddedImageVec);

    float **paddedImage = vector2float(paddedImageVec, 0);
    float *outputImage = (float *)malloc(rows * cols * sizeof(float));

    vector<int> sizes = get_sizes(paddedImageVec);

    float *devicePaddedImage;
    float *dev_out_img;    

    cudaError_t err = cudaMalloc(&devicePaddedImage, sizes[0] * sizes[1] * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }
    cudaError_t err2 = cudaMalloc(&dev_out_img, rows * cols * sizeof(float));
    if (err2 != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    cout << "main rows: " << rows << " ; cols: " << cols << endl;
    cout << "padd rows: " << sizes[0] << " ; cols: " << sizes[1] << endl;

        // Copy data to device array
    for (int i = 0; i < height; i++) {
        cudaMemcpy(&devicePaddedImage[i * width], h_data[i], width * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dev_pad, paddedImage, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);

    //  define 2d block and grid size
    int threadDim = findSqaureNum(MAX_THREADS);
    dim3 blockSize(threadDim, threadDim);
    dim3 gridSize((rows + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    cout << "Block size: " << blockSize.x << "x" << blockSize.y << endl;
    cout << "Grid size: " << gridSize.x << "x" << gridSize.y << endl;

    myKernel<<<gridSize, blockSize>>>(data, width, height);

    // // CREATE MULTIPLE STREAMS HERE
    // int num_streams = 16;
    // cudaStream_t streams[num_streams];
    // for (int i = 0; i < num_streams; i++)
    // {
    //     cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    // }
    // // for (int i = 0; i < rows; i++)
    // // {
    // // cout<<"Starting kernel call"<<endl;
    // for (int i = 0; i < rows; i++)
    // {
    //     pixel_kernel_call<<<1, 128, 0, streams[i % num_streams]>>>(dev_pad, dev_out_img, cols, halfWindowSize, halfSearchWindowSize, h, rows, i);
    // }

    // cudaDeviceSynchronize();
    // cudaMemcpy(outputImage, dev_out_img, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // vector<vector<float>> new_outputImage = intImage(outputImage, rows, cols);

    // cout << "Saving Image" << endl;

    cv::Mat dst;
    // dst = Vec2Mat(new_outputImage, "outputImage.png");

    return dst;
}

// check device properties
void checkDevice()
{
    int device = 0;
    int max_blocks_per_multiprocessor;
    int num_multiprocessors;
    int max_blocks_total;
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, device);
    cudaDeviceGetAttribute(&max_blocks_per_multiprocessor, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    cudaDeviceGetAttribute(&num_multiprocessors, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&max_blocks_total, cudaDevAttrMaxBlockDimX, device);

    int max_blocks = min(max_blocks_per_multiprocessor * num_multiprocessors, max_blocks_total);

    printf("\n---------------Device Properties---------------\n");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of blocks: %d\n", max_blocks);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
    printf("-----------------------------------------------\n\n");
}

int main(int argc, char **argv)
{

    // string image_path
    cout << "Loading image " << image_path << endl;
    cv::Mat src = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    src = resizeImage(src);
    cout << "Shape of image: " << src.size() << endl;

    checkDevice();

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