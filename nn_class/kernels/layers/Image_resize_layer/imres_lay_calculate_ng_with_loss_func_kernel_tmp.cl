
__kernel void imres_lay_calculate_ng_with_loss_func_kernel(
    __global const float *output,
    __global float *layer_res,
    __global float *next_gradients,

    const long data_size
)
{
    long i = get_global_id(0); // getting the cycle index
    if(i>=data_size)return; // checking for out of bounds of array

    next_gradients[i]=loss_gradient();


    //printf("ok \n");

}
