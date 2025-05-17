#include <include/neuron.h>
__kernel void leaky_relu_activation_activate_kernel(
    __global float *layer_res,
    const float coef,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    layer_res[i]=layer_res[i]*(layer_res[i]<=0?coef:1);

    //printf("ok \n");

}
