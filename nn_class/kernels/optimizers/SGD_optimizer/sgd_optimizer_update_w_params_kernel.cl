#include <include/neuron.h>
__kernel void sgd_optimizer_update_w_params_kernel(
    __global nn_type *weights,
    __global nn_type *gradients,
    const float learning_rate,
    const float regularization_coef,
    const long batch_size,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;
    add(&gradients[i], -regularization_coef*weights[i]*batch_size);
    add(&weights[i],learning_rate*gradients[i]);
    gradients[i]=0;
    //printf("ok\n");
}
