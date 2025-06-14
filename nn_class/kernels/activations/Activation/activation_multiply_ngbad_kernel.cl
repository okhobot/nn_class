#include <include/neuron.h>
__kernel void activation_multiply_ngbad_kernel(
    __global neuron *neurons,
    __global float *layer_res,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    neurons[i].gradient*=1;

    //printf("ok \n");
}
