#include <include/neuron.h>
__kernel void conv_lay_predict_kernel(
    __global neuron *neurons,
    __global const float *input,
    __global float *layer_res,
    __global nn_type *weights,
    const long channels_count,
    const long nx_size,
    const long ny_size,
    const long lay_res_x_size,
    const long lay_res_y_size,
    const long x_step,
    const long y_step,

    const long neurons_count,
    const long y_size,
    const long x_size
)
{
    long neuron_index = get_global_id(0); // getting the cycle index
    long y = get_global_id(1)*y_step;
    long x = get_global_id(2)*x_step;

    if(neuron_index>=neurons_count)return; // checking for out of bounds of array
    if(y+ny_size>y_size)return;
    if(x+nx_size>x_size)return;


    int res_index=neuron_index*lay_res_x_size*lay_res_y_size+y/y_step*lay_res_x_size+x/x_step;
    layer_res[res_index]=0;

    for(int channel=0;channel<channels_count;channel++)
        for(int j=0; j<ny_size;j++)
            for(int k=0; k<nx_size;k++)
            {
                layer_res[res_index]+=tof(weights[neurons[neuron_index].params_start_index+ k+j*nx_size+channel*nx_size*ny_size])*
                    input[x+k+(y+j)*x_size+channel*x_size*y_size];


            }

    //printf("ok \n");

}
