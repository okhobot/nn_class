#include <include/neuron.h>
__kernel void fcl_lay_calculate_ng_kernel(
    __global neuron *neurons,
    __global const neuron *next_neurons,
    __global const nn_type *next_weights,
    const long next_neurons_count,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    for(long n_i=0;n_i<next_neurons_count;n_i++)
        {
            neurons[i].gradient+=
            next_neurons[n_i].gradient*
            tof(next_weights[next_neurons[n_i].params_start_index+i]);
        }
    //printf("ok \n");

}
