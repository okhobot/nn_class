__kernel void max_arr_kernel(
    __global float *arr,
    __global float *res,
    const long target_index
)
{
    long i = get_global_id(0);

    float group_max = work_group_reduce_max(arr[i]);

    if (i == 0) {
        res[target_index] = group_max;
        //printf("max: %f \n",res[i]);
    }

    //printf("ok \n");

}
