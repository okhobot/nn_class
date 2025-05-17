#include <optimizers/SGD_optimizer.hpp>

SGD_optimizer::SGD_optimizer(int a_batch_size, float a_learning_rate, float a_lr_reduction_coef, float a_min_lr,float a_regularization_coef,OCLW *a_oclw) :
    Optimizer(a_learning_rate,a_lr_reduction_coef,a_min_lr,a_regularization_coef,a_oclw)
{

    km.set_default_path("kernels/optimizers/SGD_optimizer/");
    km.add_kernel("update_b_params","sgd_optimizer_update_b_params_kernel");
    km.add_kernel("update_w_params","sgd_optimizer_update_w_params_kernel");

    batch_size=a_batch_size;
}


void SGD_optimizer::update_params(Layer *layer, const nn_size_type &data_index)
{
    if((data_index+1)%batch_size!=0)return;

    weights_ptr=layer->get_weights_ptr();
    gradients_ptr=layer->get_gradients_ptr();
    neurons_ptr = layer->get_neurons_ptr();


    for(int i=0; i<layer->get_params_count(); i++)
    {
        add(&gradients_ptr[i], -regularization_coef*weights_ptr[i]*batch_size);
        add(&weights_ptr[i],learning_rate*gradients_ptr[i]);
        gradients_ptr[i]=0;
    }

    for(int i=0; i<layer->get_neurons_count(); i++)
    {
        neurons_ptr[i].b_grad-= regularization_coef*neurons_ptr[i].b*batch_size;
        add(&neurons_ptr[i].b,learning_rate*neurons_ptr[i].b_grad);
        neurons_ptr[i].b_grad=0;
    }

}


void SGD_optimizer::update_params_oclw(Layer *layer, const nn_size_type &data_index)
{
    if((data_index+1)%batch_size!=0)return;

    oclw->process_oclw(km.get("update_w_params"), {layer->get_weights_key(), layer->get_gradients_key()}, {learning_rate,regularization_coef}, {batch_size,layer->get_params_count()},layer->get_params_count());
    oclw->process_oclw(km.get("update_b_params"), {layer->get_neurons_key()}, {learning_rate,regularization_coef}, {batch_size, layer->get_neurons_count()},layer->get_neurons_count());

}

