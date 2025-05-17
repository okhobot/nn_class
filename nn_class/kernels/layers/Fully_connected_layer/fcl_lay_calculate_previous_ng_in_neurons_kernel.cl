#include <include/neuron.h>
__kernel void fcl_lay_calculate_previous_ng_in_neurons_kernel(
    __global neuron *previous_neurons,
    __global const neuron *neurons,
    __global const nn_type *weights,
    const long neurons_count,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    previous_neurons[i].gradient=0;
    for(int j=0;j<neurons_count;j++)
        {
            previous_neurons[i].gradient+=
                neurons[j].gradient*
                tof(weights[neurons[j].params_start_index+i]);
        }

    //printf("ok \n");

}
