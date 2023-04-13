

__global__ void pixel_kernel_call(float* paddedImage, float* outputImage, int cols, int windowSize, int searchWindowSize, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int halfWindowSize = windowSize / 2;
    int halfSearchWindowSize = searchWindowSize / 2;

    float weightedSum = 0;
    float similaritySum = 0;

    for(int k=-halfSearchWindowSize; k<=halfSearchWindowSize; k++){
        for(int l=-halfSearchWindowSize; l<=halfSearchWindowSize; l++){
            float dist = 0;
            float tempWeightedSum = 0;
            float tempSimilaritySum = 0;
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

int main(){
    float *dev_pad, *dev_out_img;
    int cols, windowSize, searchWindowSize, h;

    // size_t N = 128;
    size_t pad_arr_len = paddedImage.size() * paddedImage[0].size();
		
    //create buffer on device
    cudaError_t err = cudaMalloc(&dev_pad, pad_arr_len*sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }
    cudaError_t err = cudaMalloc(&dev_out_img, pad_arr_len*sizeof(float));
    if (err != cudaSuccess){
      cout<<"Dev Memory not allocated"<<endl;
      exit(-1);
    }

    cudaMemcpy(dev_pad, paddedImage, pad_arr_len * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsperblock(16, 16);
    dim3 blockspergrid((N + threadsperblock.x - 1) / threadsperblock.x, (N + threadsperblock.y - 1) / threadsperblock.y);
    pixel_kernel_call<<<blockspergrid, threadsperblock>>>(dev_pad, dev_out_img, cols, windowSize, searchWindowSize, h);
}
