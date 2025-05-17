#include <include/neuron.h>
__kernel void sigmoid_activation_multiply_ngbad_kernel(
    __global neuron *neurons,
    __global float *layer_res,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    neurons[i].gradient*=layer_res[i]*(1-layer_res[i]);

    //printf("ok \n");

}
