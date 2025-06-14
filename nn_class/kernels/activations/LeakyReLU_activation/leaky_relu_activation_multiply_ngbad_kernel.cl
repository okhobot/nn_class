#include <include/neuron.h>
__kernel void leaky_relu_activation_multiply_ngbad_kernel(
    __global neuron *neurons,
    __global float *layer_res,
    float coef,
    long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    neurons[i].gradient*=(layer_res[i]>0?1:coef);

    //printf("ok \n");
}
