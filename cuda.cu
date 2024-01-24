#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>


void mergesort(long*, long, dim3, dim3);
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);


void generateRandomNumbers(long* data, long size) {
    std::srand(std::time(NULL)); 

    for (long i = 0; i < size; ++i) {
        data[i] = std::rand() % 100; 
    }
}

#define min(a, b) (a < b ? a : b)

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;
        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}


void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    cudaMalloc((void**) &D_data, size * sizeof(long));
    cudaMalloc((void**) &D_swp, size * sizeof(long));

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);

    //
    // Copy the thread / block info to the GPU as well
    //
    cudaMalloc((void**) &D_threads, sizeof(dim3));
    cudaMalloc((void**) &D_blocks, sizeof(dim3));

    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);
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

    long size = 1000;
    long* data = new long[size];
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
