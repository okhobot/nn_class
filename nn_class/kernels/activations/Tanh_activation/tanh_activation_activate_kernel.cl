#include <include/neuron.h>
__kernel void tanh_activation_activate_kernel(
    __global float *layer_res,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    layer_res[i]=tanh(layer_res[i]);

    //printf("ok \n");

}
