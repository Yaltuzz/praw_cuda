#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>


void mergesort(int*, int, dim3, dim3);
__global__ void gpu_mergesort(int*, int*, int, int, int, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(int*, int*, int, int, int);


void generateRandomNumbers(int* data, int size) {
    std::srand(std::time(NULL)); 

    for (int i = 0; i < size; ++i) {
        data[i] = std::rand() % 100; 
    }
}

#define min(a, b) (a < b ? a : b)

__device__ void gpu_bottomUpMerge(int* source, int* dest, int start, int middle, int end) {
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

__global__ void gpu_mergesort(int* source, int* dest, int size, int width, int slices, dim3* threads, dim3* blocks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int start = width*idx*slices, 
         middle, 
         end;

    for (int slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;
        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}


void mergesort(int* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    int* D_data;
    int* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    cudaMalloc((void**) &D_data, size * sizeof(int));
    cudaMalloc((void**) &D_swp, size * sizeof(int));

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    //
    // Copy the thread / block info to the GPU as well
    //
    cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));

    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    int* A = D_data;
    int* B = D_swp;

    int nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        int slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 128;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 64;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    int size = 1000;
    int* data = new int[size];
    generateRandomNumbers(data, size);

    mergesort(data, size, threadsPerBlock, blocksPerGrid);

    bool is_sorted = true;
    for (int i = 0; i < size - 1; i++) {
        if (data[i] > data[i + 1]) {
            is_sorted = false;
            break;
        }
    }
    std::cout << "Array is sorted: " << (is_sorted ? "true" : "false") << std::endl;
}
