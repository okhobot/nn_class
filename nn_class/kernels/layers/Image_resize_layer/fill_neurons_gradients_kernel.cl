#include <include/neuron.h>
__kernel void fill_neurons_gradients_kernel(
    __global neuron *neurons,
    const float value,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    neurons[i].gradient=value;

    //printf("ok \n");

}
