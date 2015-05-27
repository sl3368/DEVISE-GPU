all: sgd_devise sgd_devise_batch sgd_devise_batch_chunk

sgd_devise_batch_chunk: sgd_devise_batch_chunk.cu
	nvcc sgd_devise_batch_chunk.cu -o sgd_devise_batch_chunk -arch=sm_30

sgd_devise_batch: sgd_devise_batch.cu
	nvcc sgd_devise_batch.cu -o sgd_devise_batch -arch=sm_30

sgd_devise: sgd_devise.cu
	nvcc sgd_devise.cu -o sgd_devise -arch=sm_30

clean: 
	rm sgd_devise sgd_devise_batch sgd_devise_batch_chunk


