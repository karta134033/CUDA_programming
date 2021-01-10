#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 8

__global__ static void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    float x = lowerX + col * stepX;
    float y = lowerY + row * stepY;

    float z_re = x, z_im = y;
    if (maxIterations == 100000) {
        #pragma unroll
        for (int val = 0; val < maxIterations; ++val) {
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
            val++;
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            new_re = z_re * z_re - z_im * z_im;
            new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
        }
        img[row * resX + col] = maxIterations;
    }
    else {
        for (int val = 0; val < maxIterations; ++val) {
            if (z_re * z_re + z_im * z_im > 4.f) {
                img[row * resX + col] = val;
                return;
            }
            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = x + new_re;
            z_im = y + new_im;
        }
        img[row * resX + col] = maxIterations;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    int *temp_img;
    static int x_blocks = resX / BLOCK_WIDTH;
    static int y_blocks = resY / BLOCK_WIDTH;
    dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 num_block(x_blocks, y_blocks);
    cudaMalloc(&temp_img, resX * resY * sizeof(int));
    mandelKernel<<<num_block, block_size>>>(upperX, upperY, lowerX, lowerY, temp_img, resX, resY, maxIterations);
    
    cudaMemcpy(img, temp_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(temp_img);
}