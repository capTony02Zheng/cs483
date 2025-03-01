#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 32
__constant__ float Mask[8000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K) refer to filter
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // change Mask for constant memory or mask for input pointer

    // Insert your GPU convolution kernel code here
    /*
    cpu version
    for (int b=0;b<B;b++) {//b = 0 .. Batch                     // for each image in the batch 
        for (int m=0;m<M;m++){ //m = 0 .. Map_out                 // for each output feature maps
            for (int h=0;h<H_out;h++)//h = 0 .. Height_out        // for each output element
                for (int w=0;w<W_out;w++)//w = 0 .. Width_out 
                {
                    // output[b][m][h][w] = 0;
                    we use code below for parallel
                    out_4d(b,m,h,w) = 0;
                    for (int c=0;c<C;c++){//c = 0 .. Channel   // sum over all input feature maps
                        for (int p=0;p<K;p++){//p = 0 .. K // KxK filter
                            for (int q=0;q<K;q++) {//q = 0 .. K
                                // output[b][m][h][w] += input[b][c][h * S + p][w * S + q] * mask[m][c][p][q]
                                out_4d(b,m,h,w) += in_4d(b,c,h * S + p,w * S + q) * mask_4d(m,c,p,q);
                            }
                        }
                    }
                }
        }
    }
    */
    int tx = threadIdx.x; // out fea x
    int ty = threadIdx.y; // out fea y
    int bx = blockIdx.x; // B
    int by = blockIdx.y; // M
    int bz = blockIdx.z; // out Height Dimesion * Width Dimesion index linearlized

    int num_block_perrow = (W_out - 1)/TILE_WIDTH +1;
    int wout_id = (bz % num_block_perrow) * TILE_WIDTH+ tx;
    int hout_id = (bz / num_block_perrow) * TILE_WIDTH+ ty;
    if (hout_id < H_out && wout_id < W_out){
        float sum = 0;
        for (int c=0;c<C;c++){//c = 0 .. Channel   // sum over all input feature maps
            for (int p=0;p<K;p++){//p = 0 .. K // KxK filter
                for (int q=0;q<K;q++) {//q = 0 .. K
                    // output[b][m][h][w] += input[b][c][h * S + p][w * S + q] * mask[m][c][p][q]
                    // out_4d(b,m,h,w) += in_4d(b,c,h * S + p,w * S + q) * mask_4d(m,c,p,q);
                    sum+=in_4d(bx,c,hout_id * S+p,wout_id * S+q) * mask_4d(by,c,p,q);
                }
            }
        }
        out_4d(bx,by,hout_id,wout_id) = sum;
    }




    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K) refer to filter
    S - stride step length
    */
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));
    cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(Mask, host_mask, (C * M * K * K)*sizeof(float));
    // use up or below
    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    float H_out = (float)(H - K)/S + 1;
    float W_out = (float)(W - K)/S + 1;
    dim3 dimGrid(B, M, ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
    // std::cout<<  "B "<< B<<  "M "<< M<<  "C "<< C<<  "H "<< H <<  "W "<< W<<  "K "<< K<< "S "<< S <<std::endl;
    // std::cout<< "dimgridz dimension " << ceil(H_out/TILE_WIDTH) << " " << ceil(W_out/TILE_WIDTH) <<std::endl;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMemcpy(host_output, device_output, (B * M * H_out * W_out)*sizeof(float), cudaMemcpyDeviceToHost);

   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask); // commment when use constant
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
