#pragma once

#include <activations/Activation.hpp>

#include <cmath>

class ReLU_activation: public Activation
{

public:

    ReLU_activation()
    {
        km.set_default_path("kernels/activations/ReLU_activation/");
        km.add_kernel("activate", "relu_activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "relu_activation_multiply_ngbad_kernel");
    }

    void activate(float *layer_res, nn_size_type neuron_count) override
    {
        for(int i=0; i<neuron_count; i++)
        {
            layer_res[i]=(layer_res[i]<=0?0:layer_res[i]);
        }
    }

    void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res, nn_size_type neuron_count) override
    {
        for(int i=0; i<neuron_count; i++)
        {
            neurons[i].gradient*=(layer_res[i]>0);
        }
    }

};
