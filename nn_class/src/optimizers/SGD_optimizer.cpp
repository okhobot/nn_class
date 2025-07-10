#include <optimizers/SGD_optimizer.hpp>

SGD_optimizer::SGD_optimizer(int a_batch_size, float a_learning_rate, float a_lr_reduction_coef, float a_regularization_coef, float a_min_lr, float a_max_lr)
{

    km.set_default_path("kernels/optimizers/SGD_optimizer/");
    km.add_kernel("update_b_params","sgd_optimizer_update_b_params_kernel");
    km.add_kernel("update_w_params","sgd_optimizer_update_w_params_kernel");

    batch_size=a_batch_size;

    learning_rate=a_learning_rate;

    regularization_coef=a_regularization_coef;

    if(a_lr_reduction_coef==0)
        debug_utils::call_warning("Optimizer init","invalid value","lr_reduction_coef cannot be 0, to ensure that the learning_rate does not decrease, set the value of lr_reduction_coefficient to 1");
    lr_reduction_coef=a_lr_reduction_coef;
    min_lr=a_min_lr;
    max_lr=a_max_lr;
}


void SGD_optimizer::update_params(Layer *layer, const nn_size_type &data_index)
{
    if((data_index+1)%batch_size!=0)return;

    weights_ptr=layer->get_weights_ptr();
    gradients_ptr=layer->get_gradients_ptr();
    neurons_ptr = layer->get_neurons_ptr();


    for(int i=0; i<layer->get_params_count(); i++)
    {
        add(&gradients_ptr[i], -2*learning_rate*regularization_coef*weights_ptr[i]);
        add(&weights_ptr[i],learning_rate*gradients_ptr[i]/batch_size);
        gradients_ptr[i]=0;
    }

    for(int i=0; i<layer->get_neurons_count(); i++)
    {
        neurons_ptr[i].b_grad-= 2*learning_rate*regularization_coef*neurons_ptr[i].b;
        add(&neurons_ptr[i].b,learning_rate*neurons_ptr[i].b_grad/batch_size);
        neurons_ptr[i].b_grad=0;
    }

}


void SGD_optimizer::update_params_oclw(Layer *layer, const nn_size_type &data_index)
{
    if((data_index+1)%batch_size!=0)return;

    oclw_ptr->process_oclw(km.get("update_w_params"), {layer->get_weights_key(), layer->get_gradients_key()}, {learning_rate,regularization_coef}, {batch_size,layer->get_params_count()},layer->get_params_count());
    oclw_ptr->process_oclw(km.get("update_b_params"), {layer->get_neurons_key()}, {learning_rate,regularization_coef}, {batch_size, layer->get_neurons_count()},layer->get_neurons_count());

}

