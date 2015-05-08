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

__global__ void single_image_global_gpu (float *image_vec, int *tr, float *W, 
                                    float *word_vecs, 
                                    int word_vecs_count,
                                    float *M_v,
                                    float *gradient,
                                    float momentum,
                                    float step_rate){

	//For each chunk
    for(int r=0; r<20; r++){
       
    	//doing everything by row
    	int block=blockIdx.x;
    	int n=threadIdx.x;

		//compute initial low-dimensional image vector Mv
    	int dot_sum=0.0;
    	__shared__ float Mv[300];

    	for ( int i=0; i<4096; i++){
    	    int idx=n*4096 + i;
    	    dot_sum+=W[idx]*image_vec[r*40960+block*4096+i];
    	}   

    	Mv[n]=dot_sum;
    	
		//initialize shared memory   
    	__shared__ float label_word_vec[300];
    	label_word_vec[n]=word_vecs[300*tr[r*10+block]+n];
    	
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

					//calculate loss
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
    	if (n_loss>0) scale=1000/n_loss;	//Hard coding right now, but should change

    	__syncthreads();

    	label_word_vec[n]+=sum_w_err[n]*n_loss; //label_word_vec is now error vector

    	//calculate outer product
    	//can move calculation and gradient step into the same loop
    	for(int j=0;j<4096; j++){
    	    atomicAdd(&W[n*4096+j],-1.0*scale*label_word_vec[n]*image_vec[j]*step_rate+momentum);
    	}

		__syncthreads();

    }
}

int main (int argc, char *argv[])
{

	// Number of input images
	int N = 20;
	// Number of validation images
    int M = N/4;   

	// Image vectors in N * 4096 float array
	float images[N*4096];
	float validation_images[M*4096];

	// Corresponding image labels
	int labels[N];
	int validation_labels[M];

	// Word vector array
	float host_word_vecs[1000*300];

	// Read input images from file
    FILE *fp;
    fp = fopen(argv[1],"r");

    char newline;

    for(int i=0;i<N;i++) {
        for(int j=0;j<4096;j++) {
            fscanf(fp, "%f%c", (images+4096*i+j), &newline);
        }
    }

    fclose(fp);

    // Read validation images from file
    fp = fopen(argv[2],"r");

    for(int i=0;i<M;i++) {
        for(int j=0;j<4096;j++) {
            fscanf(fp, "%f%c", (validation_images+4096*i+j),&newline);
        }
    }

    fclose(fp);

    //Read labels from file
    fp = fopen(argv[3],"r");
       
    for(int i=0;i<N;i++) {
            fscanf(fp, "%d%c", (labels+i), &newline);
    }   

    fclose(fp);
	
	//Read validation labels from file
    fp = fopen(argv[4],"r");

    for(int i=0;i<M;i++) {
            fscanf(fp, "%d%c", (validation_labels+i), &newline);
    }

    fclose(fp);

    // Read word vectors from file
    fp = fopen(argv[5],"r");

    for(int i=0;i<1000;i++) {
        for(int j=0;j<300;j++) {
            fscanf(fp, "%f%c", (host_word_vecs+300*i+j), &newline);
        }
    }

    fclose(fp);

	// create timers
	cudaEvent_t     start, stop;
    float           elapsedTime;

    // start the timers
    GPU_CHECKERROR( cudaEventCreate( &start ) );
    GPU_CHECKERROR( cudaEventCreate( &stop ) );

	// initialize host weight matrix (4096*300)
	float host_W[4096*300];

	// initialize weight matrix (4096*300)
	float *W;
	GPU_CHECKERROR(
		cudaMalloc((void**) &W, 4096*300*sizeof(float))
	);
	cudaMemset ((void *) W, 0, 4096*300*sizeof (unsigned int));

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

	int minibatch_size=20*10;

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

	// Gradients of minibatch of images
	float *gradients;
	GPU_CHECKERROR(
		cudaMalloc((void**) &gradients, minibatch_size * 300 * 4096 * sizeof(int))
	);

	// Low dimensional image vector Mv
	float *Mv;
	GPU_CHECKERROR(
		cudaMalloc((void**) &Mv, 300 * sizeof(float))
	);

	cudaStream_t    stream0, stream1;
	GPU_CHECKERROR( cudaStreamCreate( &stream0 ) );	
	GPU_CHECKERROR( cudaStreamCreate( &stream1 ) );	
	int num_epochs=1;

	GPU_CHECKERROR( cudaEventRecord( start, 0 ) );
	
	//For ith epoch (i.e. ith run over data )
	for(int i=0;i<num_epochs;i++) {
		//For jth image 
		for(int j=0;j<N;j+= minibatch_size*2) {
				//create chunk for images and labels
				float *img_vec_chunk_0=images+(4096*j);							//image chunk for stream0
				float *img_vec_chunk_1=images+(4096*j)+minibatch_size*4096;		//image chunk for stream1
				int *img_labels_chunk_0=labels+j;								//label chunk for stream0	
				int *img_labels_chunk_1=labels+j+minibatch_size;				//label chunk for stream1	
				
				//first stream of image and vector chunks to GPU
			    GPU_CHECKERROR ( cudaMemcpyAsync ((void *) image_vecs, (void *) img_vec_chunk_0,					
										minibatch_size* 4096 * sizeof (float),
    			            			cudaMemcpyHostToDevice,
										stream0) );
				
			    GPU_CHECKERROR ( cudaMemcpyAsync ( (void *) tr, (void *) img_labels_chunk_0,
										minibatch_size * sizeof (int),
    			            			cudaMemcpyHostToDevice,
										stream0) );
				
				//run kernel
				single_image_global_gpu<<<10, 300, 0, stream0>>>
                                        (image_vecs,							//image vectors on GPU
                                        tr,										//true labels 
                                        W,										//weight matrix
										word_vecs,								//word vectors for all 1000 classes
										1000,									//number of classes
										Mv,										//low dimensional image vector
										gradients,								//gradients of mini-batch of images
										.9,										//momentum
										.0001);									//step_rate

				//second stream of image and vector chunks to GPU
			    GPU_CHECKERROR ( cudaMemcpyAsync ((void *) image_vecs, (void *) img_vec_chunk_1,					
										minibatch_size* 4096 * sizeof (float),
    			            			cudaMemcpyHostToDevice,
										stream1) );
				
		    	GPU_CHECKERROR ( cudaMemcpyAsync ( (void *) tr, (void *) img_labels_chunk_1,
										minibatch_size * sizeof (int),
    			            			cudaMemcpyHostToDevice,
										stream1) );
				
				//run kernel
				single_image_global_gpu<<<10, 300, 0, stream1>>>
                                        (image_vecs,							//image vectors on GPU
                                        tr,										//true labels 
                                        W,										//weight matrix
										word_vecs,								//word vectors for all 1000 classes
										1000,									//number of classes
										Mv,										//low dimensional image vector
										gradients,								//gradients of mini-batch of images
										.9,										//momentum
										.0001);									//step_rate

		}

		// Pull out weights after each epoch and calculate validation accuracy
        GPU_CHECKERROR ( cudaMemcpyAsync ( (void *) &host_W, (void *) W,
                            4096*300* sizeof (float),
                            cudaMemcpyHostToDevice,
                            stream1) );

        // Calculate validation accuracy here
 
	}

	GPU_CHECKERROR( cudaStreamSynchronize( stream0 ) );
	GPU_CHECKERROR( cudaStreamSynchronize( stream1 ) );

	//Time the kernel run
	GPU_CHECKERROR( cudaEventRecord( stop, 0 ) );

    GPU_CHECKERROR( cudaEventSynchronize( stop ) );
    GPU_CHECKERROR( cudaEventElapsedTime( &elapsedTime,
                start, stop ) );

    printf( "Time taken:  %3.1f ms\n", elapsedTime );

	//Free device memory
	GPU_CHECKERROR( cudaFree( W ) );
	GPU_CHECKERROR( cudaFree( word_vecs ) );
	GPU_CHECKERROR( cudaFree( image_vecs ) );
	GPU_CHECKERROR( cudaFree( tr ) );
	GPU_CHECKERROR( cudaFree( gradients ) );
	GPU_CHECKERROR( cudaFree( Mv ) );

	//Destroy streams
	GPU_CHECKERROR( cudaStreamDestroy( stream0 ) );
	GPU_CHECKERROR( cudaStreamDestroy( stream1 ) );

 
}
