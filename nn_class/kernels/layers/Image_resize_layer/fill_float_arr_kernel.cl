__kernel void fill_float_arr_kernel(
    __global float *arr,
    const float value,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;

    arr[i]=value;

    //printf("ok \n");

}
