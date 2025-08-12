#include <include/neuron.h>
__kernel void dense_lay_calculate_previous_ng_in_neurons_kernel(
    __global neuron *previous_neurons,
    __global const neuron *neurons,
    __global const nn_type *weights,
    const long neurons_count,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    float res=0;
    for(int j=0;j<neurons_count;j++)
        {
            res+=
                neurons[j].gradient*
                tof(weights[neurons[j].params_start_index+i]);
        }

    previous_neurons[i].gradient=res;

    //printf("ok \n");

}
