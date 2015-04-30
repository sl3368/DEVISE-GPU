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
 
__global__ void single_image_global_gpu (float *image_vec, int tr, float *W, 
									float *word_vecs, 
									int word_vecs_count,
									float *Mv,
									float *gradient,
									float momentum,
									float step_rate){
	
	//doing everything by row
	int n=threadIdx.x;
	int dot_sum=0.0;
	for ( int i=0; i<4096; i++){
		int idx=n*4096 + i;
		dot_sum+=W[idx]*image_vec[i];
	}
	Mv[n]=dot_sum;
	
	__shared__ float label_word_vec[300];
	label_word_vec[n]=word_vecs[300*tr+n];
	
	
	__shared__ float w_label_Mv[1];
	w_label_Mv[0]=0.0;
	atomicAdd(&w_label_Mv[0],Mv[n]*label_word_vec[n]);

	__shared__ float sum_w_err[300];
	sum_w_err[n]=0.0;	
	
	__syncthreads();

	int n_loss=0;
	float loss=0.0;
		

	if(n==1){
		for(int i=0; i<word_vecs_count; i++){
			if(i!=tr){
				
				//calculate w_j_Mv
				int offset=i*300;
				float w_j_Mv=0.0;
				for(int j=0; j<300; j++)
					w_j_Mv+=Mv[j]+word_vecs[offset+j];

				float loss_j = 0.1 - w_label_Mv[0] + w_j_Mv; //hard coding of margin
				if(loss_j>0){
                			n_loss++;
                			loss += loss_j;
			                for(int k=0;k<300;k++)
						sum_w_err[k] += word_vecs[offset+k];
					//breaking
					i=word_vecs_count;
				}	
			}
		}
	}
	//scaling loss
	float scale=0.0;
	if (n_loss>0)
		scale=1000/n_loss;//Hard coding right now, but should change
	

	__syncthreads();
	label_word_vec[n]+=sum_w_err[n]*n_loss; //label_word_vec is now error vector
	

	//calculate outer product
	//can move calculation and gradient step into the same loop
	for(int j=0;j<4096; j++){
		gradient[n*4096+j]=scale*label_word_vec[n]*image_vec[j];
	}

	//updating with momentum coefficient (momentum * step)
	for(int j=0;j<4096; j++){
		W[n*4096+j]-=gradient[n*4096+j]*step_rate+momentum;
	}
	
}

int main (int argc, char *argv[])
{

   //1. Need to get data and word2vec in correct format:
	int N = 50000;
	//1-image vectors in 50,000 * 4096 float array
	float images[N][4096];
	//2-Corresponding image label
	float labels[N];

	// How do we get word vectors? From a pickle file?
	float host_word_vecs[1000][300];
	//3-check if the label has a word vector, if not, throw out 

	//creating data
	for(int i=0;i<N;i++){
		for(int k=0; k<4096; k++){
			images[i][k]=3.0;
		}
		labels[i]=2.0;
	}
	
	for(int i=0; i<1000; i++){
		for(int k=0; k<300; k++){
			host_word_vecs[i][k]=5.0;
		}
	}

	
	// initialize weight matrix (4096*300)
	float *W;
	// put on global memory of the device
	GPU_CHECKERROR(
		cudaMalloc((void**) &W, 4096*300*sizeof(float))
	);

	//NEED TO MEMSET TO ZERO!!

	// weve used up 4.91 MB	of global memory

	//put word_vec matrix  (1000 * 300)
	//onto device global memory

	float *word_vecs;
	GPU_CHECKERROR(
		cudaMalloc((void**) &word_vecs, 1000 * 300 * sizeof(float))
	);
	GPU_CHECKERROR(
    		cudaMemcpy ((void *) word_vecs,
                (void *) host_word_vecs,
                1000 * 300 * sizeof (unsigned int),
                cudaMemcpyHostToDevice)
    	); 

	// weve used up 4.91 + 1.2 = 6.11 MB

	int num_epochs=10;
	int minibatch_size=1;

	// Container for minibatch of images on device
	float *image_vecs; 	
	GPU_CHECKERROR(
		cudaMalloc((void**) &image_vecs, minibatch_size * 4096 * sizeof(float))
	);

	// True labels for the minibatch of images
	int *tr;
	GPU_CHECKERROR(
		cudaMalloc((void**) &tr, minibatch_size * sizeof(int))
	);

	// Mv
	float *Mv;
	GPU_CHECKERROR(
		cudaMalloc((void**) &Mv, 300 * sizeof(float))
	);

//	float img_vec_chunk[minibatch_size][4096];
//	float img_label_chunk[minibatch_size][1];	
	
	// for e in epochs:
	for(int i=0;i<num_epochs;i++) {
		//for n in total_images/minibatch_size:
		for(int j=0;j<ceil(N/(2*minibatch_size)); j++) {
			//Using streams:
				//create chunk for image_vecs and labels
				float *img_vec_chunk=images[0]+(4096*j);
				float *img_labels_chunk=labels+j;						
				
				//load onto GPU
				GPU_CHECKERROR(
		    			cudaMemcpy ((void *) image_vecs,(void *) img_vec_chunk, minibatch_size*4096 * sizeof (float),
    				            cudaMemcpyHostToDevice)
				);
				
				GPU_CHECKERROR(
		    			cudaMemcpy ((void *) tr,(void *) img_labels_chunk, minibatch_size * sizeof (float),
    				            cudaMemcpyHostToDevice)
				);
				
				//run kernel

				//create second chunk

				//load onto GPU

				//run kernel

				//perform validation somehow




				
		}
	}

/**
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
 
 
    // how much time has elapsed?
    gettimeofday (&t1, 0);
 
    //
    // GPU version
//
 
 
    // make sure the GPU is finished doing everything!
    cudaDeviceSynchronize();
 
    // finish timing:
    gettimeofday (&t2, 0);
 
    // complete the timing:
    float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    float timdiff2 = (1000000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)) / 1000000.0;
 **/
 
    printf("ending\n");
 
 
}
