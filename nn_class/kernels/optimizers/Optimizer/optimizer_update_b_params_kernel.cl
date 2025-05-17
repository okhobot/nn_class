#include <include/neuron.h>
__kernel void optimizer_update_b_params_kernel(
    __global neuron *neurons,
    const float learning_rate,
    const float regularization_coef,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array
    neurons[i].b_grad-=regularization_coef*neurons[i].b;
    add(&neurons[i].b, learning_rate*neurons[i].b_grad);
    neurons[i].b_grad=0;

    //printf("ok \n");
}
