#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <stdlib.h> 
 
// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
 
 
//need multiple kernels for different types of dot products
//and outer products mainly

 
// same as above, only for GPU: cannot return values, so must store
// result in global memory location ("count")
// also: must make sure this thread maps to useful data! (what if
// the # of threads is > than the number of data elements!)
__global__ void primeP_gpu (unsigned int max, unsigned int *A, unsigned int *count)
{
 
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    printf("Thread: %d",n); 
    // do nothing if we are not in the useable space of
    // threads (see kernel launch call: you may be creating
    // more threads than you need)
    if (n >= max) return;
 
    unsigned int a = A[n];
    int found=0;
    for (int i=2; i*i<=a; i++) {
        if (a % i == 0){
               found=1;
        }
    }
 
    if (found == 0) {
        // don't do this: threads overwrite each other's values
        // causing undercount:
        //    *count = *count + 1;
 
        // instead, use atomic operations:
        atomicAdd(count, 1);
    }
 
}
 
 
int main (int argc, char *argv[])
{

   //1. Need to get data and word2vec in correct format:
	//1-image vectors in 500,000 * 4096 float array
	//2-Corresponding image label
	//3-check if the label has a word vector, if not, throw out
	//(resultng in img_vecs (n*4096), img_labels (n,1), word_vecs (n,300)
	// n is the number of filtered image vectors
	
	// initialize weight matrix (4096*300)

	// initialize gradient matrix (4096 * 300)

	//for some n iterations:

		//compute gradient of weights

		//step=gradient * step_rate * momentum

		//add step to weights

		//print out some validation if possible

	
	//Compute Gradient:
		



    //Simple error checking
    if(argc<3 || argc>4){
	printf("ERROR: Usage ./primeV filename number_of_integers number_of_threads(optional)\n");
	exit(EXIT_FAILURE);
    }
     
    printf("beginning\n");
 
    struct timeval t0, t1, t2;
 
    //Filename to read in:
    char* filename=argv[1];

    FILE* f=fopen(filename,"r");
    if( f == NULL ){
      perror("Error on file open.\n");
      exit(EXIT_FAILURE);
    }
 
    // How many integers are in the test file:
    unsigned int numIntegers = 1000000;
    if (sscanf(argv[2], "%i", &numIntegers)!=1){
	printf("Second argument must be the number of integers in file!\n");
    	exit(EXIT_FAILURE);
    }

    //Number of threads, defaults to 512 if not specified
    unsigned int numThreads=512;
    if(argc==4){
	if(sscanf(argv[3], "%i", &numThreads)!=1) {
		printf("Third argument must be number of threads per block\n");
		exit(EXIT_FAILURE);
	}
    }
 
    // start basic timing:
    gettimeofday (&t0, 0);
 
    // allocate the array to hold the data:
    unsigned int *h_intAArray;
    h_intAArray = (unsigned int *) malloc (numIntegers * sizeof (unsigned int));
 
    // read file for integers
    int number=0;
    for(int i=0;i<numIntegers; i++) {
	fscanf (f, "%d", &number); 
        h_intAArray[i] = number;
    }
     
    // count how many are prime:
    unsigned int primeCount = 0;
    for (int i = 0; i < numIntegers; ++i) {
        int isprime = primeP(h_intAArray, i);
        primeCount += isprime;
    }
 
    // how much time has elapsed?
    gettimeofday (&t1, 0);
 
    //
    // GPU version
    //
 
    // allocate the A array on the GPU, and copy the data over:
    unsigned int *d_intAArray;
 
    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_intAArray, numIntegers * sizeof (unsigned int))
    );
 
    GPU_CHECKERROR(
    cudaMemcpy ((void *) d_intAArray,
                (void *) h_intAArray,
                numIntegers * sizeof (unsigned int),
                cudaMemcpyHostToDevice)
    );
 
    // allocate a location to hold the count, and set it to zero:
    unsigned int *d_numprimes;
    cudaMalloc ((void **) &d_numprimes, sizeof (unsigned int));
    cudaMemset ((void *) d_numprimes, 0, sizeof (unsigned int));
 
 
    // we want to run a grid of 512-thread blocks (for reasons you
    // will understand later. How many such blocks will we need?
    // NOTE: be SURE to prevent integer division if you use this
    // snippet: that "1.0*" is absolutely required to prevent
    // rounding before the ceil() call:
    unsigned int threads_per_block;
    if (numThreads<1){
	threads_per_block = 512;
    } else {
	threads_per_block = numThreads;
    }
    unsigned int num_blocks = ceil (numIntegers / (1.0*threads_per_block) );
    printf("Using %d blocks, each with %d threads.\n",num_blocks, threads_per_block); 

    // launch the kernel:
    primeP_gpu<<<num_blocks, threads_per_block>>>
                                        (numIntegers,
                                        d_intAArray,
                                        d_numprimes);
 
    // get back the count:
    unsigned int h_numprimes;
 
    cudaMemcpy ((void *) &h_numprimes,
                (void *) d_numprimes,
                sizeof(unsigned int),
                cudaMemcpyDeviceToHost);
 
    // make sure the GPU is finished doing everything!
    cudaDeviceSynchronize();
 
    // finish timing:
    gettimeofday (&t2, 0);
 
    // free up the memory:
    cudaFree (d_intAArray);
    cudaFree (d_numprimes);
    free (h_intAArray); 
 
    // complete the timing:
    float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    float timdiff2 = (1000000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)) / 1000000.0;
 
    //printf ("done: time taken for serial version is %3.1f s\n", timdiff1);
    //printf ("done: time taken for parallel version is %3.1f s\n", timdiff2);
 
    //printf ("serial version found this many primes:%d \n", primeCount);
    //printf ("parallel version found this many primes:%d \n", h_numprimes);

    //print the result as specified
    printf("%d %3.1f %d %3.1f\n",primeCount, timdiff1, h_numprimes, timdiff2);
 
    printf("ending\n");
 
 
}
