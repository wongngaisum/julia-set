#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

#define CUDA_CHECK( err ) (cuda_checker(err, __FILE__, __LINE__))

static void cuda_checker( cudaError_t err, const char *file, int line ) {

	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
    
}

struct cppComplex {

	float r; 
	float i;
	__host__ __device__ cppComplex( float a, float b ) : r(a), i(b) {}
	__host__ __device__ float magnitude2( void ) {
		return r * r + i * i;
	}
	__host__ __device__ cppComplex operator*( const cppComplex& a ) {
		return cppComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__host__ __device__ cppComplex operator+( const cppComplex& a ) {
		return cppComplex(r + a.r, i + a.i);
	}

};

__host__ __device__ int julia( int x, int y ) {

	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
	
	cppComplex c(-0.8, 0.156);
	cppComplex a(jx, jy);

	int i = 0;
	for(i = 0; i < 200; i++){
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;

}

void julia_set_cpu() {

	unsigned char *pixels = new unsigned char[DIM * DIM]; 

	for (int x = 0; x < DIM; ++x) {
		for (int y = 0; y < DIM; ++y) {
			pixels[x + y * DIM] = 255 * julia(x, y);
		}
	}

	FILE *f = fopen("julia_cpu.ppm", "wb");

    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            fputc(pixels[(y * DIM + x)], f);
            fputc(0, f);
            fputc(0, f);
		}
    }
    fclose(f);

    delete [] pixels;

}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
/*Begin the GPU part*/
///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////// 

__global__ void kernel( unsigned char *ptr ) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;	// row index
	int y = threadIdx.y + blockIdx.y * blockDim.y;	// col index

	int tid = y + x * DIM;	// left to right
		
	if (x < DIM && y < DIM)	// prevent out of bound access
		ptr[tid] = 255 * julia(y, x);

}

void julia_set_gpu() {

	unsigned char *pixels = new unsigned char[DIM * DIM]; // for host
	unsigned char *dev_bitmap;	// for device 

	CUDA_CHECK(cudaMalloc((void**)&dev_bitmap, DIM * DIM * sizeof(unsigned char)));	// allocate memory on gpu

	CUDA_CHECK(cudaMemcpy(dev_bitmap, pixels, DIM * DIM * sizeof(unsigned char), cudaMemcpyHostToDevice));	// copy memory to device

	dim3 threads_per_block = dim3(32, 32);
	dim3 blocks_per_grid = dim3((DIM + threads_per_block.x - 1) / threads_per_block.x, (DIM + threads_per_block.y - 1) / threads_per_block.y);

	kernel<<<blocks_per_grid, threads_per_block>>>(dev_bitmap);	// execute

	CUDA_CHECK(cudaMemcpy(pixels, dev_bitmap, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost));	// copy memory to host

	// write to file
	FILE *f = fopen("julia_gpu.ppm", "wb");
    fprintf(f, "P6\n%i %i 255\n", DIM, DIM);
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            fputc(pixels[(y * DIM + x)], f);   // 0 .. 255
            fputc(0, f);
            fputc(0, f);
      }
    }

    fclose(f);

    // free memory
    CUDA_CHECK(cudaFree(dev_bitmap));
	delete [] pixels; 

}

int main( void ) {
	
	float time;
	cudaEvent_t start, stop;

	// record cpu execution time

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start, 0));

	julia_set_cpu();

	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate using CPU:  %3.1f ms \n", time);	

	// record gpu execution time

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start, 0));

	julia_set_gpu();

	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate using GPU:  %3.1f ms \n", time);	

	// flush buffer
	cudaDeviceReset();

}