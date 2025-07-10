__kernel void softmax_sum_exp_kernel(
    __global float *arr,
    __global float *res,
    const long max_val_index,
    const long target_res_index
)
{
    long i = get_global_id(0);

    arr[i]=exp(arr[i] - res[max_val_index]);
    float group_sum = work_group_reduce_add(arr[i]);

    if (i == 0) {
        res[target_res_index] = group_sum;
        //printf("sum: %f \n",res[i]);
    }

    //printf("ok \n");

}
