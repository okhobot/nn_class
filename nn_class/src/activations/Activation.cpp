#include <activations/Activation.hpp>


void Activation::activate_oclw(std::string layer_res_key, const nn_size_type &neuron_count)
{
    oclw->process_oclw(km.get("activate"), {layer_res_key}, {}, {neuron_count},neuron_count);
}


void Activation::multiply_neuron_gradient_by_activation_derivative_oclw(std::string neurons_key,std::string layer_res, const nn_size_type &neuron_count)
{
    oclw->process_oclw(km.get("multiply_ng_by_activation_derivative"), {neurons_key,layer_res}, {}, {neuron_count},neuron_count);
}
