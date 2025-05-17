__kernel void test_kernel(
    __global const float *data,
    long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    printf("ok %f ", data[i]);

}
