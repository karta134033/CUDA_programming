#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 32
#define GROUP_OF_PIXELS 1

__global__ void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_OF_PIXELS;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    for (int i = 0; i < GROUP_OF_PIXELS && col < resX; i++, col++) {
        float x = lowerX + col * stepX;
        float y = lowerY + row * stepY;
        float z_re = x, z_im = y;
        int val = 0;
        for (; val < maxIterations; ++val) {
            if (z_re * z_re + z_im * z_im > 4.f) break;
            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
        }
        img[row * resX + col] = val;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    int *temp_img; 
    int *output;
    size_t pitch;
    cudaHostAlloc((void **)&output, resX * resY * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch((void **) &temp_img, &pitch, resX * sizeof (int), resY);
    
    int group_block_size = BLOCK_SIZE * GROUP_OF_PIXELS;
    static int x_blocks = resX % group_block_size == 0 ? resX / group_block_size : resX / group_block_size + 1;
    static int y_blocks = resY % BLOCK_SIZE == 0 ? resY / BLOCK_SIZE : resY / BLOCK_SIZE + 1;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_block(x_blocks, y_blocks);
    mandelKernel<<<num_block, block_size>>>(upperX, upperY, lowerX, lowerY, temp_img, resX, resY, maxIterations);

    cudaMemcpy(output, temp_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, output, resX * resY * sizeof(int));
    cudaFreeHost(output);
    cudaFree(temp_img);
}