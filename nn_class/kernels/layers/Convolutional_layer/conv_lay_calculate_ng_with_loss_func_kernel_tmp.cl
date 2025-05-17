#include <include/neuron.h>
__kernel void conv_lay_calculate_ng_with_loss_func_kernel(
    __global neuron *neurons,
    __global const float *output,
    __global float *layer_res,
    __global float *converted_layer_res,
    const long neuron_out_size,
    const long data_size
)
{
    long neuron_index = get_global_id(0); // getting the cycle index
    if(neuron_index>=data_size)return; // checking for out of bounds of array

    neurons[neuron_index].gradient=0;
    converted_layer_res[neuron_index]=0;
    for(int i=neuron_index*neuron_out_size;i<(neuron_index+1)*neuron_out_size;i++)
    {
        neurons[neuron_index].gradient+=loss_gradient();
        converted_layer_res[neuron_index]+=layer_res[i];
    }


    neurons[neuron_index].gradient/=neuron_out_size;
    converted_layer_res[neuron_index]/=neuron_out_size;

    //printf("ok \n");

}
