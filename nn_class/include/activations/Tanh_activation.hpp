#pragma once

#include <activations/Activation.hpp>

#include <cmath>

class Tanh_activation: public Activation
{

public:

    Tanh_activation()
    {
        km.set_default_path("kernels/activations/Tanh_activation/");
        km.add_kernel("activate", "tanh_activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "tanh_activation_multiply_ngbad_kernel");
    }

    void activate(float *layer_res, nn_size_type neuron_count) override
    {
        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            layer_res[i] = std::tanh(layer_res[i]);
        }
    }


    void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res, nn_size_type neuron_count) override
    {
        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            neurons[i].gradient *= 1.0f - layer_res[i] * layer_res[i];
        }
    }

};

