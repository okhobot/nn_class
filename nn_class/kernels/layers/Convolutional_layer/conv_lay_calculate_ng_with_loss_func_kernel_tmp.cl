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
    long neuron_index = get_global_id(0);
    if(neuron_index>=data_size)return;

    float clres=0, ngres=0;
    for(int i=neuron_index*neuron_out_size;i<(neuron_index+1)*neuron_out_size;i++)
    {
        ngres+=loss_gradient();
        clres+=layer_res[i];
    }


    neurons[neuron_index].gradient=ngres/neuron_out_size;
    converted_layer_res[neuron_index]=clres/neuron_out_size;

    //printf("ok \n");

}
