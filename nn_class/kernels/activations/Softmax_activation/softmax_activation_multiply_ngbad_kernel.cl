#include <include/neuron.h>
__kernel void softmax_activation_multiply_ngbad_kernel(
    __global neuron *neurons,
    __global float *layer_res,
    __global float *tmp_res,
    const long tmp_res_index
)
{
    long i = get_global_id(0);

    neurons[i].gradient = layer_res[i] * (neurons[i].gradient - tmp_res[tmp_res_index]);

    //printf("ok \n");

}
