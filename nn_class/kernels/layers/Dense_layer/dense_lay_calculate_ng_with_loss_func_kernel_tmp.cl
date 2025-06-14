#include <include/neuron.h>
__kernel void dense_lay_calculate_ng_with_loss_func_kernel(
    __global neuron *neurons,
    __global const float *output,
    __global float *layer_res,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    neurons[i].gradient=loss_gradient();


    //printf("ok \n");

}
