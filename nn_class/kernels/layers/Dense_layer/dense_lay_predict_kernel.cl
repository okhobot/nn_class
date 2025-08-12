#include <include/neuron.h>
__kernel void dense_lay_predict_kernel(
    __global neuron *neurons,
    __global const float *input,
    __global float *layer_res,
    __global nn_type *weights,
    const long params_count,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    long pstart_index=neurons[i].params_start_index;
    float sum=tof(neurons[i].b);

    for(long j=0;j<params_count;j++)
        sum+=tof(weights[pstart_index+j])*input[j];

    layer_res[i]=sum;

    //printf("ok \n");

}
