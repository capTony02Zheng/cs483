#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"
//
#define TILE_WIDTH 20
__constant__ float Mask[8000];
__constant__ __half Mask_h[8000];
cudaStream_t stream1, stream2, stream3;

/*
Tiled shared memory convolution (2 points)
Shared memory matrix multiplication and input matrix unrolling (3 points)
Kernel fusion for unrolling and matrix-multiplication (requires previous optimization) (2 points)
Weight matrix (kernel values) in constant memory (0.5 point)
Tuning with restrict and loop unrolling (considered as one optimization only if you do both) (3 points)
Sweeping various parameters to find best values (block sizes, amount of thread coarsening) (0.5 point)
Multiple kernel implementations for different layer sizes (1 point)
Input channel reduction: tree (3 point)
Input channel reduction: atomics (2 point)
FP16 arithmetic. (note this can modify model accuracy slightly) (4 points)
Using Streams to overlap computation with data transfer (4 points)
An advanced matrix multiplication algorithm (register-tiled, for example) (5 points)
Using Tensor Cores to speed up matrix multiplication (5 points)
Overlap-Add method for FFT-based convolution (note this is very hard, and may not yield a large performace increase due to mask size) (8 points)
*/
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
    C - number of input feature maps // num of input colors (like rgb)
    M - number of output feature maps // The number of filters or kernels applied during the convolution operation. Each filter produces one feature map in the output.
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K) // one filter size
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
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
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

////
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
////

///done
__global__ void conv_forward_kernel_st(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S, int stream_offset)
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
    C - number of input feature maps // num of input colors (like rgb)
    M - number of output feature maps // The number of filters or kernels applied during the convolution operation. Each filter produces one feature map in the output.
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K) // one filter size
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // change Mask for constant memory or mask for input pointer

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x; // out fea x
    int ty = threadIdx.y; // out fea y
    int bx = blockIdx.x + stream_offset; // B ?????
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
/// to be done
__global__ void conv_forward_kernel_ts(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    // int sml = TILE_WIDTH + K - 1;
    const int sml = 3;
    // __shared__ float share_m[3][3];
    

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int tx = threadIdx.x; // out fea x
    int ty = threadIdx.y; // out fea y
    int bx = blockIdx.x; // B
    int by = blockIdx.y; // M
    int bz = blockIdx.z; // out Height Dimesion * Width Dimesion index linearlized

    int num_block_perrow = (W_out - 1)/TILE_WIDTH +1;
    int wout_id = (bz % num_block_perrow) * TILE_WIDTH+ tx;
    int hout_id = (bz / num_block_perrow) * TILE_WIDTH+ ty;
    //(bz % num_block_perrow) * TILE_WIDTH is the row of the tile
    //(bz / num_block_perrow) * TILE_WIDTH is the column of the tile


    float sum = 0;
    for (int c=0;c<C;c++){//c = 0 .. Channel   // sum over all input feature maps
        for (int row = wout_id; row < (bz % num_block_perrow) * TILE_WIDTH + sml; row++) {
            for (int col = hout_id; col < (bz / num_block_perrow) * TILE_WIDTH+ sml;col++) {

            }
        }
        for (int p=0;p<K;p++){//p = 0 .. K // KxK filter
            for (int q=0;q<K;q++) {//q = 0 .. K
                sum+=in_4d(bx,c,hout_id * S+p,wout_id * S+q) * mask_4d(by,c,p,q);
            }
        }
    }
    out_4d(bx,by,hout_id,wout_id) = sum;


    ////
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


    
    // float sum = 0;
    // for (int c=0;c<C;c++){//c = 0 .. Channel   // sum over all input feature maps

    //     for(int i = hout_id; i < bz / num_block_perrow * TILE_WIDTH + sml; i += TILE_WIDTH){

    //         for(int j = wout_id; j < bz % num_block_perrow * TILE_WIDTH + sml; j += TILE_WIDTH){
    //             if(i < H && j < W){
    //                 share_m[i - bz % num_block_perrow * TILE_WIDTH][j - bz / num_block_perrow * TILE_WIDTH] = in_4d(bx, c, i, j);
    //             }else{
    //                 share_m[i - bz % num_block_perrow * TILE_WIDTH][j - bz / num_block_perrow * TILE_WIDTH] = 0.0f;
    //             }
    //         }
    //     }
    //     __syncthreads();

    //     if (hout_id < H_out && wout_id < W_out){
    //         for (int p=0;p<K;p++){//p = 0 .. K // KxK filter
    //             for (int q=0;q<K;q++) {//q = 0 .. K
    //                 if(tx < TILE_WIDTH && ty < TILE_WIDTH){
    //                     // sum+=in_4d(bx,c,hout_id * S+p,wout_id * S+q) * mask_4d(by,c,p,q); // out_4d(b,m,h,w) += in_4d(b,c,h * S + p,w * S + q) * mask_4d(m,c,p,q);
    //                     sum += share_m[tx+p][ty+q] * mask_4d(by,c,p,q);
    //                 }
    //             }
    //         }
    //     }
    //     __syncthreads();
    // }
    // if(hout_id < H_out && wout_id < W_out){
    //   out_4d(bx, by, hout_id, wout_id) = sum;
    // }


    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

////done
__global__ void conv_forward_kernel_fp16(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{


    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    __half* ou = (__half*)output;
    __half* in = (__half*)input;
    __half* ma = (__half*)mask;

    #define out_4d(i3, i2, i1, i0) ou[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) in[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) ma[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // change Mask for constant memory or mask for input pointer

    int tx = threadIdx.x; // out fea x
    int ty = threadIdx.y; // out fea y
    int bx = blockIdx.x; // B
    int by = blockIdx.y; // M
    int bz = blockIdx.z; // out Height Dimesion * Width Dimesion index linearlized

    int num_tile_per_row = (W_out - 1)/TILE_WIDTH +1; // math: is W_out / TILE_WIDTH, those 1 just for round up
    int wout_id = (bz % num_tile_per_row) * TILE_WIDTH+ tx;
    int hout_id = (bz / num_tile_per_row) * TILE_WIDTH+ ty;
    if (hout_id < H_out && wout_id < W_out){
        __half sum = __float2half(0.0f);
        for (int c=0;c<C;c++){//c = 0 .. Channel   // sum over all input feature maps
            for (int p=0;p<K;p++){//p = 0 .. K // KxK filter
                for (int q=0;q<K;q++) {//q = 0 .. K
                    sum = __hfma(in_4d(bx,c,hout_id * S+p,wout_id * S+q),mask_4d(by,c,p,q),sum);
                }
            }
        }
        out_4d(bx,by,hout_id,wout_id) = sum;
    }




    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

////done
__global__ void in_unroll(const float *input, float *output, const int M, const int C, const int H, const int W, const int K, const int S){
    
    
    // dim3 dimGridUnroll(B ,ceil(1.0 * W_out * H_out * C / TILE_WIDTH), 1);
    // unroll matrix input = (K*K*C) * (H_out * W_out)
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define out_3d(i2, i1, i0) output[(i2) * (C* K * K* H_out * W_out) + (i1) * (H_out * W_out) + i0]

    // dim3 DimGrid2(ceil(1.0 * numCColumns/TILE_WIDTH), ceil(1.0 * numCRows /TILE_WIDTH),1);
    // dim3 DimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    int hout_id = blockIdx.y * blockDim.y + threadIdx.y; // the row index of the mout_position
    int wout_id = blockIdx.x * blockDim.x + threadIdx.x; // the col index of the mout_position
    int bz = blockIdx.z;

    if (hout_id < H_out && wout_id < W_out) {
        for (int c = 0; c < C; c++) {
            int w_unroll = hout_id * W_out + wout_id;
            int w_base = c * (K*K);
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++){
                    // calculate the vertical matrix index
                    int h_unroll = w_base + p * K + q;
                    out_3d(bz, h_unroll, w_unroll) = in_4d(bz, c, hout_id * S + p, wout_id * S + q);// problem with here
                }
            }
        }
    }

    #undef out_3d
    #undef in_4d
}

////done
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,const int Batch, const int M, const int Channel, const int H, const int W, const int K,const int S) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //   int bc = blockIdx.c // B
  // A is mask B is unrolled
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int q = 0; q < ((numBRows-1)/TILE_WIDTH + 1); q++){
        if(row < numARows && (q * TILE_WIDTH + tx) < numAColumns){
            subTileM[ty][tx] = A[row * numAColumns + (q * TILE_WIDTH + tx)];
        } else {
            subTileM[ty][tx] = 0;
        }

        if(col < numBColumns && (q * TILE_WIDTH + ty < numBRows)){
            subTileN[ty][tx] = B[bz * numBRows*numBColumns+ col + (q * TILE_WIDTH + ty) * numBColumns];
        } else {
            subTileN[ty][tx] = 0;
        }

        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += subTileM[ty][i] * subTileN[i][tx];
        }

        __syncthreads();

        if(row < numARows && col < numBColumns) {
            C[bz * numCRows * numCColumns + row * numBColumns + col] = Pvalue;
        }

    }

    //correct below


    // int bx = blockIdx.x;
    // int by = blockIdx.y;
    // int bz = blockIdx.z;

    // int tx = threadIdx.x;
    // int ty = threadIdx.y;

    // int bdx = blockDim.x;
    // int bdy = blockDim.y;
    // int Row = by*bdy+ty;
    // int Col = bx*bdx+tx;
    // if((Row < numCRows) && (Col < numCColumns)) {
    //     float Pvalue = 0;
    //     for (int k = 0; k < numAColumns; k++)
    //     Pvalue += A[Row * numAColumns + k] * B[bz*numBRows*numBColumns+ k * numBColumns + Col];
    //     C[bz*numCRows*numCColumns+Row * numCColumns + Col] = Pvalue;
    // }
}





__host__ void base_pro(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{   
    std::cout<< "baseline applied" <<std::endl;
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));
    cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_mask, (C * M * K * K)*sizeof(float));
    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void fp16_pro(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{   
    std::cout<< "fp16 applied" <<std::endl;
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    __half *tmp1;
    tmp1 = (__half*)malloc(B * C * H * W * sizeof(__half));
    for (int i = 0; i < B * C * H * W; i++) {
        tmp1[i] = __float2half(host_input[i]);
    }
    cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(__half));
    cudaMemcpy(*device_input_ptr, tmp1, (B * C * H * W)*sizeof(__half), cudaMemcpyHostToDevice);
    free(tmp1);

    __half *tmp2;
    tmp2 = (__half*)malloc(C * M * K * K * sizeof(__half));
    for (int i = 0; i < C * M * K * K; i++) {
        tmp2[i] = __float2half(host_mask[i]);
    }
    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(__half));
    cudaMemcpy(*device_mask_ptr, tmp2, (C * M * K * K)*sizeof(__half), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(Mask_h, tmp2, (C * M * K * K)*sizeof(__half));
    free(tmp2);
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(__half));
}

__host__ void stream_all(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{   
    std::cout<< "stream applied" <<std::endl;
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));
    cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(float));
    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);

    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    int bsize = (B+2) / 3;
    int s1s = 0;//stream 1 start
    int s1e = min(bsize,B);
    int s2s = s1e;
    int s2e = min(s2s+bsize,B);
    int s3s = s2e;
    int s3e = B;
    // float* device_in = (float*)malloc((B * C * H * W)*sizeof(float));
    // device_in = *device_input_ptr;
    if (s1s < s1e) {
        cudaMemcpyAsync((void*)(*device_input_ptr), (void*)host_input, 
        (s1e - s1s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream1);
    }
    if (s2s < s2e) {
        cudaMemcpyAsync(((void*)*device_input_ptr) + s2s*C*H*W*sizeof(float), ((void*)host_input)+ s2s*C*H*W*sizeof(float), 
        (s2e - s2s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream2);
    }
    if (s3s < B) {
        cudaMemcpyAsync(((void*)*device_input_ptr) + s3s*C*H*W*sizeof(float), ((void*)host_input)+ s3s*C*H*W*sizeof(float), 
        (s3e - s3s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream3);
    }
    //kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    if (s1s < s1e) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid1(s1e-s1s,M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid1,dimBlock,0,stream1>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, 0);
    }
    if (s2s < s2e) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid2(s2e-s2s,M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid2,dimBlock,0,stream2>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, s2s);
    }
    if (s3s < B) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid3(s3e-s3s, M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid3,dimBlock,0,stream3>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, s3s);
    }
    //copy back
    if (s1s < s1e) {
        cudaMemcpyAsync(((void*)host_output) + s1s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s1s*M*H_out*W_out*sizeof(float), 
        (s1e - s1s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream1);
    }
    if (s2s < s2e) {
        cudaMemcpyAsync(((void*)host_output) + s2s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s2s*M*H_out*W_out*sizeof(float), 
        (s2e - s2s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream2);
    }
    if (s3s < B) {
        cudaMemcpyAsync(((void*)host_output) + s3s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s3s*M*H_out*W_out*sizeof(float), 
        (s3e - s3s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream3);
    }
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}

__host__ void m_unroll_pro(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{   
    std::cout<< "m_unroll applied" <<std::endl;
    float H_out = (float)(H - K)/S + 1;
    float W_out = (float)(W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));

    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);

    float *device_input_ptr_before_unroll;
    cudaMalloc((void **) &device_input_ptr_before_unroll, (B * C * H * W)*sizeof(float));
    cudaMemcpy(device_input_ptr_before_unroll, host_input, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) device_input_ptr, B * K * K * C * ceil(H_out) * ceil(W_out) * sizeof(float)); // now as unroll
    dim3 dimGridUnroll(ceil(1.0 * H_out/TILE_WIDTH), ceil(1.0 * W_out /TILE_WIDTH),B);
    dim3 dimBlockUnroll(TILE_WIDTH, TILE_WIDTH, 1);
    in_unroll<<<dimGridUnroll, dimBlockUnroll>>>(device_input_ptr_before_unroll, *device_input_ptr, M, C, H, W, K, S);
    cudaFree(device_input_ptr_before_unroll);
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // std::cout << "conv_forward_gpu_prolog start" << std::endl;
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // call corrsesponding handler
    base_pro(host_output, host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr,B,M, C, H, W,  K, S);
    // fp16_pro(host_output, host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr,B,M, C, H, W,  K, S);
    // m_unroll_pro(host_output, host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr,B,M, C, H, W,  K, S);
    // stream_all(host_output, host_input, host_mask, device_output_ptr, device_input_ptr, device_mask_ptr,B,M, C, H, W,  K, S); // this funtion handle all
    
   
    // all below are scratch!!!!!!!!!!
    
    // matrix unrolled and tiled multiply done
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{   
    
    // std::cout << "conv_forward_gpu start" << std::endl;
    // Set the kernel dimensions and call the kernel
    
    float H_out = (float)(H - K)/S + 1;
    float W_out = (float)(W - K)/S + 1;
    dim3 dimGrid(B, M, ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S); // normal
    // conv_forward_kernel_fp16<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S); // fp16
    
    
    // Shared memory matrix multiplication and input matrix unrolling (3 points):
    // int numCRows = M;
    // int numCColumns =floor(H_out) * floor(W_out);
    // dim3 DimGrid2(ceil(1.0 * numCColumns/TILE_WIDTH), ceil(1.0 * numCRows /TILE_WIDTH), B);
    // dim3 DimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    // matrixMultiplyShared<<<DimGrid2,DimBlock2>>>(device_mask, device_input, device_output, M, (K*K*C), (K*K*C), numCColumns ,numCRows, numCColumns, B, M, C, H, W, K, S);
    //Shared memory matrix multiplication done

    // stream start:
    






    //scratch below
    //// unroll matrix filter = M * (K*K*C)
    //// unroll matrix input = (K*K*C) * (H_out * W_out)

    /*
    int numCRows = M;
    int numCColumns =floor(H_out) * floor(W_out);
    // printf("cumcrow %d ,numccol %d\n", numCRows, numCColumns);
    float *unrolled;
    cudaMalloc((void **) &unrolled, B * K * K * C * ceil(H_out) * ceil(W_out) * sizeof(float));
    dim3 dimGridUnroll(ceil(1.0 * H_out/TILE_WIDTH), ceil(1.0 * W_out /TILE_WIDTH),B);
    dim3 dimBlockUnroll(TILE_WIDTH, TILE_WIDTH, 1);
    in_unroll<<<dimGridUnroll, dimBlockUnroll>>>(device_input, unrolled, M, C, H, W, K, S);
    float* unroll_host = (float*)malloc(B * K * K * C * H_out * W_out * sizeof(float));
    cudaMemcpy(unroll_host, unrolled, B * K * K * C * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    */
    // std::cout << std::endl;
    // for (int i = 0; i < (K*K*C);i++) {
    //     printf(" %f", unroll_host[i * numCColumns + 4440]);
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < K;i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf(" %f", device_mask[i * K + j]);
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < K*K;i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf(" %f", unrolled[i * K*K + j]);
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < H;i++) {
    //     for (int j = 0; j < W; j++) {
    //         printf(" %f", device_input[i * W + j]);
    //     }
    //     std::cout << std::endl;
    // }
    

    // cudaFree(unrolled);
    //// Shared memory matrix multiplication done

    /**/


    
} 


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    // std::cout << "conv_forward_gpu_epilog start" << std::endl;
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // normal
    cudaMemcpy(host_output, device_output, (B * M * H_out * W_out)*sizeof(float), cudaMemcpyDeviceToHost); // comment when use fp16

    //fp16
    // __half *tmp3;
    // tmp3 = (__half*)malloc(B * M * H_out * W_out * sizeof(__half));
    // cudaMemcpy(tmp3, device_output, (B * M * H_out * W_out)*sizeof(__half), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < B * M * H_out * W_out; i++){
    //     host_output[i] = __half2float(tmp3[i]);
    // }
    // free(tmp3);
    //fp16 done
    
   
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask); // commment when use constant
}






//// none of my business below
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

// develop code in prolog
/*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    C - number of input feature maps // num of input colors (like rgb)
    M - number of output feature maps // The number of filters or kernels applied during the convolution operation. Each filter produces one feature map in the output.
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K) // one filter size
    S - stride step length
    */
    // baseline start:
    // const int H_out = (H - K)/S + 1;
    // const int W_out = (W - K)/S + 1;
    // cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));
    // cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(float));
    // cudaMemcpy(*device_input_ptr, host_input, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(Mask, host_mask, (C * M * K * K)*sizeof(float));
    // cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    // cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);
    // normal done

    // for (int i = 0; i < K;i++) {
    //     for (int j = 0; j < K; j++) {
    //         for (int q = 0; q < C;q++) {
    //             printf(" %f", host_mask[i * K + j]);
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    // for (int i = 0; i < K * K * C;i++) {
    //     printf(" %f", host_mask[i]);
    // }
    // std::cout << std::endl;
    // cmkk refer to that for every input color(in total C) in one image(one of B)
    // it will apply to M filter to get to the corresponding output position in output feature map(image), thus total C*M, and each filter size K * K, so CMKK.
    // all corresponding output position in output feature map received sum to one of the final output feature map.

    // fp16:
    
    // __half *tmp1;
    // tmp1 = (__half*)malloc(B * C * H * W * sizeof(__half));
    // for (int i = 0; i < B * C * H * W; i++) {
    //     tmp1[i] = __float2half(host_input[i]);
    // }
    // cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(__half));
    // cudaMemcpy(*device_input_ptr, tmp1, (B * C * H * W)*sizeof(__half), cudaMemcpyHostToDevice);
    // free(tmp1);
    // __half *tmp2;
    // tmp2 = (__half*)malloc(C * M * K * K * sizeof(__half));
    // for (int i = 0; i < C * M * K * K; i++) {
    //     tmp2[i] = __float2half(host_mask[i]);
    // }
    // cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(__half));
    // cudaMemcpy(*device_mask_ptr, tmp2, (C * M * K * K)*sizeof(__half), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(Mask_h, tmp2, (C * M * K * K)*sizeof(__half));
    // free(tmp2);
    // cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(__half));
    //fp16 done 


    // std::cout << "conv_forward_gpu_prolog done" << std::endl;
    ////Using Streams to overlap computation with data transfer (4 points)
    

    /*
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));
    cudaMalloc((void **) device_input_ptr, (B * C * H * W)*sizeof(float));
    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    int bsize = (B+2) / 3;
    int s1s = 0;//stream 1 start
    int s1e = min(bsize,B);
    int s2s = s1e;
    int s2e = min(s2s+bsize,B);
    int s3s = s2e;
    int s3e = B;
    // float* device_in = (float*)malloc((B * C * H * W)*sizeof(float));
    // device_in = *device_input_ptr;
    if (s1s < s1e) {
        cudaMemcpyAsync((void*)(*device_input_ptr), (void*)host_input, 
        (s1e - s1s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream1);
    }
    if (s2s < s2e) {
        cudaMemcpyAsync(((void*)*device_input_ptr) + s2s*C*H*W*sizeof(float), ((void*)host_input)+ s2s*C*H*W*sizeof(float), 
        (s2e - s2s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream2);
    }
    if (s3s < B) {
        cudaMemcpyAsync(((void*)*device_input_ptr) + s3s*C*H*W*sizeof(float), ((void*)host_input)+ s3s*C*H*W*sizeof(float), 
        (s3e - s3s)*C*H*W*sizeof(float),cudaMemcpyHostToDevice,stream3);
    }
    //kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    if (s1s < s1e) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid1(s1e-s1s,M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid1,dimBlock,0,stream1>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, 0);
    }
    if (s2s < s2e) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid2(s2e-s2s,M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid2,dimBlock,0,stream2>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, s2s);
    }
    if (s3s < B) {
        float H_out = (float)(H - K)/S + 1;
        float W_out = (float)(W - K)/S + 1;
        dim3 dimGrid3(s3e-s3s, M,ceil(H_out/TILE_WIDTH)*ceil(W_out/TILE_WIDTH));
        conv_forward_kernel_st<<<dimGrid3,dimBlock,0,stream3>>>(*device_output_ptr,*device_input_ptr,*device_mask_ptr, B, M, C, H, W, K, S, s3s);
    }
    //copy back
    if (s1s < s1e) {
        cudaMemcpyAsync(((void*)host_output) + s1s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s1s*M*H_out*W_out*sizeof(float), 
        (s1e - s1s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream1);
    }
    if (s2s < s2e) {
        cudaMemcpyAsync(((void*)host_output) + s2s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s2s*M*H_out*W_out*sizeof(float), 
        (s2e - s2s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream2);
    }
    if (s3s < B) {
        cudaMemcpyAsync(((void*)host_output) + s3s*M*H_out*W_out*sizeof(float), ((void*)*device_output_ptr)+ s3s*M*H_out*W_out*sizeof(float), 
        (s3e - s3s)*M*H_out*W_out*sizeof(float),cudaMemcpyDeviceToHost,stream3);
    }
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    */
    
    // stream done*/
    /*
    // matrix unrolled and tiled multiply:
    float H_out = (float)(H - K)/S + 1;
    float W_out = (float)(W - K)/S + 1;
    cudaMalloc((void **) device_output_ptr, (B * M * H_out * W_out)*sizeof(float));

    cudaMalloc((void **) device_mask_ptr, (C * M * K * K)*sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, (C * M * K * K)*sizeof(float), cudaMemcpyHostToDevice);

    float *device_input_ptr_before_unroll;
    cudaMalloc((void **) &device_input_ptr_before_unroll, (B * C * H * W)*sizeof(float));
    cudaMemcpy(device_input_ptr_before_unroll, host_input, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) device_input_ptr, B * K * K * C * ceil(H_out) * ceil(W_out) * sizeof(float)); // now as unroll
    dim3 dimGridUnroll(ceil(1.0 * H_out/TILE_WIDTH), ceil(1.0 * W_out /TILE_WIDTH),B);
    dim3 dimBlockUnroll(TILE_WIDTH, TILE_WIDTH, 1);
    in_unroll<<<dimGridUnroll, dimBlockUnroll>>>(device_input_ptr_before_unroll, *device_input_ptr, M, C, H, W, K, S);
    cudaFree(device_input_ptr_before_unroll);
    */