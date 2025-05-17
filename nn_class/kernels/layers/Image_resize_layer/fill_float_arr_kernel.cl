__kernel void fill_float_arr_kernel(
    __global float *arr,
    const float value,
    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    arr[i]=value;

    //printf("ok \n");

}
