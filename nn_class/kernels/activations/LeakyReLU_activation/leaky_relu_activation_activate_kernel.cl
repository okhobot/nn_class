#include <include/neuron.h>
__kernel void leaky_relu_activation_activate_kernel(
    __global float *layer_res,
    const float coef,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    layer_res[i]=layer_res[i]*(layer_res[i]<=0?coef:1);

    //printf("ok \n");

}
