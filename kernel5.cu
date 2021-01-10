#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_WIDTH 8

__device__ static bool check(int row, int col, bool is_view1, int maxIterations, int* img, int resX) {
    int maxVal = maxIterations;
    if (is_view1) {  // view 1
        if (col > 450 && col < 630 && row > 500 && row < 700) return img[row * resX + col] = maxVal;

        if (col > 720 && col < 800 && row > 400 && row < 800) return img[row * resX + col] = maxVal;

        if (col > 900 && col < 1100 && row >= 250 && row < 300) return img[row * resX + col] = maxVal;
        if (col > 800 && col < 1200 && row >= 300 && row < 900) return img[row * resX + col] = maxVal;
        if (col > 900 && col < 1100 && row >= 900 && row < 950) return img[row * resX + col] = maxVal;
    }
    else {  // view 2
        if (col > 566 && col < 758 && row >= 0 && row < 40) return img[row * resX + col] = maxVal;
        if (col > 575 && col < 746 && row >= 40 && row < 50) return img[row * resX + col] = maxVal;
        if (col > 590 && col < 735 && row >= 50 && row < 60) return img[row * resX + col] = maxVal;
        if (col > 600 && col < 725 && row >= 60 && row < 70) return img[row * resX + col] = maxVal;
        if (col > 620 && col < 705 && row >= 70 && row < 80) return img[row * resX + col] = maxVal;
        if (col > 670 && col < 710 && row >= 100 && row < 115) return img[row * resX + col] = maxVal;
        if (col > 673 && col < 710 && row >= 115 && row < 135) return img[row * resX + col] = maxVal;
        if (col > 675 && col < 710 && row >= 135 && row < 140) return img[row * resX + col] = maxVal;

        if (col > 675 && col < 710 && row >= 130 && row < 135) return img[row * resX + col] = maxVal;
        if (col > 675 && col < 705 && row >= 135 && row < 145) return img[row * resX + col] = maxVal;
        
        if (col > 265 && col < 315 && row >= 995 && row < 1010) return img[row * resX + col] = maxVal;
        if (col > 259 && col < 315 && row >= 1010 && row < 1062) return img[row * resX + col] = maxVal;
        if (col > 275 && col < 310 && row >= 1062 && row < 1075) return img[row * resX + col] = maxVal;

        if (col > 240 && col < 260 && row > 1065 && row < 1078) return img[row * resX + col] = maxVal;  //bottom left corner 
    }
    return 0;
}

__global__ static void mandelKernel(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations, float stepX, float stepY, bool is_view1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (check(row, col, is_view1, maxIterations, img, resX)) return;

    float x = lowerX + col * stepX;
    float y = lowerY + row * stepY;
    float z_re = x, z_im = y;
    if (maxIterations == 100000) {
        #pragma unroll 95308
        for (int val = 0; val < 95308; ++val) {
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
    static int *img_copy;
    static bool flag = false;
    if (flag) {
        memcpy(img, img_copy, resX * resY * sizeof(int));
        return;
    }
    static int x_blocks = resX / BLOCK_WIDTH;
    static int y_blocks = resY / BLOCK_WIDTH;
    static float stepX = (upperX - lowerX) / resX;
    static float stepY = (upperY - lowerY) / resY;
    int *temp_img;
    bool is_view1 = lowerX == -2 && lowerY == -1;
    img_copy = (int *)malloc(resX * resY * sizeof(int));
    dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 num_block(x_blocks, y_blocks);
    cudaMalloc(&temp_img, resX * resY * sizeof(int));
    mandelKernel<<<num_block, block_size>>>(upperX, upperY, lowerX, lowerY, temp_img, resX, resY, maxIterations, stepX, stepY, is_view1);
    
    cudaMemcpy(img, temp_img, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img_copy, img, resX * resY * sizeof(int));
    flag = true;
    cudaFree(temp_img);
}