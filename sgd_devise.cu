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
 
__global__ void single_image_global_gpu (float *image_vec, int *tr, float *W, 
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
	label_word_vec[n]=word_vecs[300*tr[0]+n];
	
	
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
			if(i!=tr[0]){
				
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

	int N = 20;
	//1-image vectors in 50,000 * 4096 float array
	float images[N*4096];
	//2-Corresponding image label
	int labels[N];

	printf("Here\n");	
	float host_word_vecs[1000*300];

	//3-check if the label has a word vector, if not, throw out 

	ifstream im;
	im.open(argv[1]);
	
	for(int i=0;i<N;i++) {
		for(int j=0;j<4096;j++) {
			im >> *(images+4096*i+j);
		}
	}
	
	im.close();

	ifstream wvec;
	wvec.open(argv[2]);
	
	for(int i=0;i<1000;i++) {
		for(int j=0;j<300;j++) {
			wvec >> *(host_word_vecs+300*i+j);
		}
	}
	
	wvec.close();

	//creating data
	//for(int i=0;i<N;i++){
	//	for(int k=0; k<4096; k++){
	//		images[i*4096+k]=3.0;
	//	}
	//	labels[i]=2;
	//}
	//
	//for(int i=0; i<1000; i++){
	//	for(int k=0; k<300; k++){
	//		host_word_vecs[i*300+k]=5.0;
	//	}
	//}

	cudaEvent_t     start, stop;
    float           elapsedTime;

    // start the timers
    GPU_CHECKERROR( cudaEventCreate( &start ) );
    GPU_CHECKERROR( cudaEventCreate( &stop ) );

	// initialize weight matrix (4096*300)
	float *W;
	GPU_CHECKERROR(
		cudaMalloc((void**) &W, 4096*300*sizeof(float))
	);
	cudaMemset ((void *) W, 0, 4096*300*sizeof (unsigned int));
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

	//Gradients
	//float gradients[300*minibatch_size*4096];	
	float *gradients;
	GPU_CHECKERROR(
		cudaMalloc((void**) &gradients, minibatch_size * 300 * 4096 * sizeof(int))
	);

	// Mv
	float *Mv;
	GPU_CHECKERROR(
		cudaMalloc((void**) &Mv, 300 * sizeof(float))
	);

	cudaStream_t    stream0, stream1;
	GPU_CHECKERROR( cudaStreamCreate( &stream0 ) );	
	GPU_CHECKERROR( cudaStreamCreate( &stream1 ) );	
	int num_epochs=1;

	GPU_CHECKERROR( cudaEventRecord( start, 0 ) );
	
	//For ith epoch
	for(int i=0;i<num_epochs;i++) {
		//For jth image 
		for(int j=0;j<N;j+= minibatch_size*2) {
			//Using streams:
				//create chunk for image_vecs and labels
				float *img_vec_chunk_0=images+(4096*j);
				float *img_vec_chunk_1=images+(4096*j)+minibatch_size*4096;
				int *img_labels_chunk_0=labels+j;						
				int *img_labels_chunk_1=labels+j+minibatch_size;						
				
				//load onto GPU
			    	GPU_CHECKERROR ( cudaMemcpyAsync ((void *) image_vecs, (void *) img_vec_chunk_0,					
											minibatch_size* 4096 * sizeof (float),
    				            			cudaMemcpyHostToDevice,
											stream0) );
				
			    	GPU_CHECKERROR ( cudaMemcpyAsync ( (void *) tr, (void *) img_labels_chunk_0,
											minibatch_size * sizeof (int),
    				            			cudaMemcpyHostToDevice,
											stream0) );
				
				//run kernel
				single_image_global_gpu<<<1, 300, 0, stream0>>>
                                        (image_vecs,
                                        tr,
                                        W,
					word_vecs,
					1000,
					Mv,
					gradients,
					.9,
					.0001);
				//create second chunk

			    	GPU_CHECKERROR ( cudaMemcpyAsync ((void *) image_vecs, (void *) img_vec_chunk_1,					
											minibatch_size* 4096 * sizeof (float),
    				            			cudaMemcpyHostToDevice,
											stream1) );
				
		    		GPU_CHECKERROR ( cudaMemcpyAsync ( (void *) tr, (void *) img_labels_chunk_1,
											minibatch_size * sizeof (int),
    				            			cudaMemcpyHostToDevice,
											stream1) );
				
				//run kernel
				single_image_global_gpu<<<1, 300, 0, stream1>>>
                                        (image_vecs,
                                        tr,
                                        W,
					word_vecs,
					1000,
					Mv,
					gradients,
					.9,
					.0001);
				//load onto GPU

				//run kernel

				//perform validation somehow

		}
	}

	GPU_CHECKERROR( cudaStreamSynchronize( stream0 ) );
	GPU_CHECKERROR( cudaStreamSynchronize( stream1 ) );

	GPU_CHECKERROR( cudaEventRecord( stop, 0 ) );

    GPU_CHECKERROR( cudaEventSynchronize( stop ) );
    GPU_CHECKERROR( cudaEventElapsedTime( &elapsedTime,
                start, stop ) );

    printf( "Time taken:  %3.1f ms\n", elapsedTime );

	//GPU_CHECKERROR( cudaFreeHost( images ) );
	//GPU_CHECKERROR( cudaFreeHost( labels ) );
	//GPU_CHECKERROR( cudaFreeHost( host_word_vecs ) );
	GPU_CHECKERROR( cudaFree( W ) );
	GPU_CHECKERROR( cudaFree( word_vecs ) );
	GPU_CHECKERROR( cudaFree( image_vecs ) );
	GPU_CHECKERROR( cudaFree( tr ) );
	GPU_CHECKERROR( cudaFree( gradients ) );
	GPU_CHECKERROR( cudaFree( Mv ) );

	GPU_CHECKERROR( cudaStreamDestroy( stream0 ) );
	GPU_CHECKERROR( cudaStreamDestroy( stream1 ) );
	//GPU_CHECKERROR( cudaStreamDestroy( stream1 ) );
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
