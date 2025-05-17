#include <include/neuron.h>
__kernel void fcl_lay_predict_kernel(
    __global neuron *neurons,
    __global const float *input,
    __global float *fcl_lay_res,
    __global nn_type *weights,
    const long params_count,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    fcl_lay_res[i]=tof(neurons[i].b);
    for(long j=0;j<params_count;j++)
        fcl_lay_res[i]+=tof(weights[neurons[i].params_start_index+j])*input[j];

     //printf("ok \n");

}
