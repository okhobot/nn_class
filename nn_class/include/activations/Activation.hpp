#pragma once
#include <neuron.h>
#include <oclw.hpp>
#include <Kernels_manager.hpp>
class Activation
{
protected:
    OCLW *oclw=0;
    Kernels_manager km;

public:
    Activation()
    {
        km.set_default_path("kernels/activations/Activation/");
        km.add_kernel("activate", "activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "activation_multiply_ngbad_kernel");
    }

    virtual void set_oclw(OCLW *a_oclw)
    {
        oclw=a_oclw;
    }


    virtual std::vector<std::string> get_kernels_paths()// for kernels initialization
    {
        return km.get_kernels_paths();
    }

    virtual void activate(float *layer_res, nn_size_type data_size) {}

    virtual void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res,  nn_size_type neuron_count) {}// for neurons gradients calculation


    virtual void activate_oclw(std::string layer_res_key, nn_size_type data_size);

    virtual void multiply_neuron_gradient_by_activation_derivative_oclw(std::string neurons_key,std::string layer_res_key, nn_size_type neuron_count);
};
