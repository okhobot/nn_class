#include <include/neuron.h>
__kernel void conv_lay_convert_to_neurons_kernel(
    __global neuron *neurons,
    __global float *converted_ng,
    __global float *layer_res,
    __global float *converted_layer_res,
    long neuron_out_size,
    long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    neurons[i].gradient=0;
    converted_layer_res[i]=0;
    for(int j=i*neuron_out_size;j<(i+1)*neuron_out_size;j++)
    {
        neurons[i].gradient+=converted_ng[j];
        converted_layer_res[i]+=layer_res[j];
    }
    neurons[i].gradient/=neuron_out_size;
    converted_layer_res[i]/=neuron_out_size;

    //printf("ok \n");

}
