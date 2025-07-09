#include <include/neuron.h>
__kernel void sgd_optimizer_update_b_params_kernel(
    __global neuron *neurons,
    const float learning_rate,
    const float regularization_coef,
    const long batch_size,
    const long data_size
)
{
    long i = get_global_id(0);
    if(i>=data_size)return;
    neurons[i].b_grad-=regularization_coef*neurons[i].b;
    add(&neurons[i].b, learning_rate*neurons[i].b_grad/batch_size);
    neurons[i].b_grad=0;
    //printf("ok\n");
}
