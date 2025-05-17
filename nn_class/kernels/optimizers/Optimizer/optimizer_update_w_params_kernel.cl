#include <include/neuron.h>
__kernel void optimizer_update_w_params_kernel(
    __global nn_type *weights,
    __global nn_type *gradients,
    const float learning_rate,
    const float regularization_coef,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array
    add(&gradients[i], regularization_coef*weights[i]);
    add(&weights[i], learning_rate*gradients[i]);
    gradients[i]=0;
    //printf("ok \n");
}
