#include <include/neuron.h>
__kernel void fcl_lay_calculate_gradients_with_ng_kernel(
    __global neuron *neurons,
    __global const float *input,
    __global nn_type *gradients,
    const long neurons_count,
    const long params_count
)
{
    long i = get_global_id(0); // getting the neuron index
    long inp_i=get_global_id(1);
    if(i>=neurons_count)return; // checking for out of bounds of array
    if(inp_i>=params_count)return;


    add(&gradients[i*neurons[i].params_count+inp_i], fromf(input[inp_i]*neurons[i].gradient));

    if(inp_i==0)neurons[i].b_grad+=fromf(neurons[i].gradient);

    //printf("ok \n");

}
