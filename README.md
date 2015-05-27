#**Accelerating the DeViSE Objective in CUDA**
*Sameer Lal, Prateek Goel*

To compile:
`make`

For specific types *(single image, mini-batch, or mini-batch chunks)*:
`make <type>`

To run:
`./<type> <image_vectors> <held_out_image_vectors> <image_labels> <held_out_image_labels> <word_vectors>`


The single image method computes the gradient with respect to the objective a single image at a time. 
Mini-batch moves multiple images at a time to the GPU for computation. Chunks moves multiple 
mini-batches onto the GPU for each kernel invocation.
