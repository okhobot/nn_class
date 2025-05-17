#pragma once

#include <activations/Activation.hpp>

#include <cmath>

class Sigmoid_activation: public Activation
{

public:

    Sigmoid_activation()
    {
        km.set_default_path("kernels/activations/Sigmoid_activation/");
        km.add_kernel("activate", "sigmoid_activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "sigmoid_activation_multiply_ngbad_kernel");
    }

    void activate(float *layer_res, const nn_size_type &neuron_count) override
    {
        for(int i=0; i<neuron_count; i++)
        {
            layer_res[i]=1.0/(1+exp(-layer_res[i]));

            if(_isnanf(layer_res[i]))
                debug_utils::call_error(1,"Sigmoid_activation::activate","nan error","neuron_index, neurons_res: ", {i,layer_res[i]});

        }
    }


    void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res, const nn_size_type &neuron_count) override
    {
        for(int i=0; i<neuron_count; i++)
        {
            neurons[i].gradient*=layer_res[i]*(1-layer_res[i]);
        }
    }

};

