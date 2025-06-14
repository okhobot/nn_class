#include <include/neuron.h>
__kernel void dense_lay_calculate_previous_ng_kernel(
    __global const neuron *neurons,
    __global const nn_type *weights,
    __global float *previous_gradients,
    const long neurons_count,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    previous_gradients[i]=0;
    for(int j=0;j<neurons_count;j++)
        {
            previous_gradients[i]+=
                neurons[j].gradient*
                tof(weights[neurons[j].params_start_index+i]);
        }

    //printf("ok \n");

}
