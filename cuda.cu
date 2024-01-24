#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <ctime>

#define min(a, b) (a < b ? a : b)

void mergesort(int*, int, dim3, dim3);
__global__ void gpu_mergesort(int*, int*, int, int, int, dim3*, dim3*);
__device__ void gpu_merge(int*, int*, int, int, int);

__device__ void gpu_merge(int* source, int* dest, int start, int middle, int end) {
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
        middle = min(start + (width / 2), size);
        end = min(start + width, size);
        gpu_merge(source, dest, start, middle, end);
        start += width;
    }
}


void mergesort(int* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    dim3* device_threads;
    dim3* device_blocks;
    int* device_data;
    int* device_swap;

    cudaMalloc((void**) &device_data, size * sizeof(int));
    cudaMalloc((void**) &device_swap, size * sizeof(int));
    cudaMalloc((void**) &device_threads, sizeof(dim3));
    cudaMalloc((void**) &device_blocks, sizeof(dim3));

    cudaMemcpy(device_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
  

    int* A = device_data;
    int* B = device_swap;

    int nThreads = threadsPerBlock.x * blocksPerGrid.x;

    for (int width = 2; width < (size * 2); width *= 2) {
        int slices = size / ((nThreads) * width) + 1;

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, device_threads, device_blocks);
        cudaDeviceSynchronize();

        if (A==device_data){
            A = device_swap;
        }else{
           A = device_data; 
        }
         if (B==device_data){
            B = device_swap;
        }else{
           B = device_data; 
        }
    }

    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 128;
    blocksPerGrid.x = 64;

    int size = 1000;
    int* data = new int[size];
    std::srand(std::time(NULL)); 

    for (int i = 0; i < size; ++i) {
        data[i] = std::rand() % 150; 
    }

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
