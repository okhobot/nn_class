#include <include/neuron.h>
__kernel void grad_dot_output_kernel(
    __global neuron *neurons,
    __global float *arr,
    __global float *res,
    const long target_index
)
{
    long i = get_global_id(0);

    float group_sum = work_group_reduce_add(neurons[i].gradient*arr[i]);

    if (i == target_index) {
        res[i] = group_sum;
    }

    //printf("ok \n");

}
