#include <include/neuron.h>
__kernel void fill_neurons_gradients_kernel(
    __global neuron *neurons,
    const float value,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    neurons[i].gradient=value;

    //printf("ok \n");

}
