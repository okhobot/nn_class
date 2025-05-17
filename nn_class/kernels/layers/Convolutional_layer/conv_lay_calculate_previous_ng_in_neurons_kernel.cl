#include <include/neuron.h>
__kernel void conv_lay_calculate_previous_ng_in_neurons_kernel(
    __global neuron *previous_neurons,
    __global const neuron *neurons,
    __global const nn_type *weights,
    const long neurons_count,
    const long nx_size,
    const long ny_size,
    const long x_size,
    const long y_size,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    previous_neurons[i].gradient=0;
    for(int j=0; j<neurons_count; j++)
    {
            for(int x=0; x<min(min(x_size-i%x_size, i%x_size+1),nx_size);x++)
                for(int y=0; y<min(min(y_size-i/x_size, i/x_size+1),ny_size);y++)
                {
                    previous_neurons[i].gradient+=
                        neurons[j].gradient*
                        tof(weights[neurons[j].params_start_index+x+y*nx_size+i/(x_size*y_size)*nx_size*ny_size]);
                }
    }


    //printf("ok \n");

}

