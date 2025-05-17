#include <include/neuron.h>
__kernel void conv_lay_calculate_gradients_with_ng_kernel(
    __global neuron *neurons,
    __global const float *input,
    __global nn_type *gradients,

    const long neurons_count,
    const long x_size,
    const long y_size,

    const long lay_res_x_size,
    const long lay_res_y_size,
    const long x_step,
    const long y_step,



    const long channels_count,
    const long ny_size,
    const long nx_size
)
{
    long channel = get_global_id(0); // getting the cycle index
    long j = get_global_id(1);
    long k = get_global_id(2);

    if(channel>=channels_count)return; // checking for out of bounds of array
    if(j>ny_size)return;
    if(k>nx_size)return;



    for(int neuron_index=0; neuron_index<neurons_count; neuron_index++)
        for(int y=0;y+ny_size<=y_size;y+=y_step)
            for(int x=0;x+nx_size<=x_size;x+=x_step)
            {
                add(&gradients[neurons[neuron_index].params_start_index+ k+j*nx_size+channel*nx_size*ny_size],

                input[x+k+(y+j)*x_size+channel*x_size*y_size]
                *fromf(neurons[neuron_index].gradient ) );
            }
    //printf("ok \n");

}

