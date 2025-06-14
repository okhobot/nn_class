#include <include/neuron.h>
__kernel void softmax_activation_activate_kernel(
    __global float *layer_res,
    __global float *tmp_arr,
    const long tmp_arr_index,
    const long data_size
)
{
    long i = get_global_id(0);


    layer_res[i]/=tmp_arr[tmp_arr_index];

    //printf("ok \n");

}
