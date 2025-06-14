#include <include/neuron.h>
__kernel void relu_activation_activate_kernel(
    __global float *layer_res,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    layer_res[i]=layer_res[i]<=0?0:layer_res[i];

    //printf("ok \n");
}
