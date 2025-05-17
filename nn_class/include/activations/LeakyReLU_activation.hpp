#pragma once

#include <activations/Activation.hpp>

#include <cmath>

class LeakyReLU_activation: public Activation
{
protected:
    float coef;

public:

    LeakyReLU_activation(float a_coef=0.25)
    {
        coef=a_coef;

        km.set_default_path("kernels/activations/LeakyReLU_activation/");
        km.add_kernel("activate", "leaky_relu_activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "leaky_relu_activation_multiply_ngbad_kernel");
    }

    void activate(float *layer_res, const nn_size_type &neuron_count)override
    {
        for(int i=0; i<neuron_count; i++)
        {
            layer_res[i]=layer_res[i]*(layer_res[i]<=0?coef:1);
        }
    }


    void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res, const nn_size_type &neuron_count) override
    {
        for(int i=0; i<neuron_count; i++)
        {
            neurons[i].gradient*=(layer_res[i]>0?1:coef);
        }
    }

    void activate_oclw(std::string layer_res_key, const nn_size_type &neuron_count)override
    {
        oclw->process_oclw(km.get("activate"), {layer_res_key}, {coef}, {neuron_count},neuron_count);
    }


    void multiply_neuron_gradient_by_activation_derivative_oclw(std::string neurons_key,std::string layer_res, const nn_size_type &neuron_count)override
    {
        oclw->process_oclw(km.get("multiply_ng_by_activation_derivative"), {neurons_key,layer_res}, {coef}, {neuron_count},neuron_count);
    }
};
