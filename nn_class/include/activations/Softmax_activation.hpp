#pragma once

#include <activations/Activation.hpp>

#include <cmath>

class Softmax_activation: public Activation
{
    float sum_tmp, max_tmp;
    std::string tmp_res_key;
public:

    Softmax_activation()
    {
        km.set_default_path("kernels/activations/Softmax_activation/");
        km.add_kernel("activate", "softmax_activation_activate_kernel");
        km.add_kernel("multiply_ng_by_activation_derivative", "softmax_activation_multiply_ngbad_kernel");

        km.add_kernel("sum_exp", "softmax_sum_exp_kernel");
        km.add_kernel("max", "max_arr_kernel");
        km.add_kernel("grad_dot_output", "grad_dot_output_kernel");

        tmp_res_key="softmax_activation_tmp_res";
    }

    void set_oclw(OCLW *a_oclw_ptr) override
    {
        oclw_ptr=a_oclw_ptr;
        std::vector<float> tmp_res(2,0);

        oclw_ptr->add_and_write_variable(tmp_res_key,CL_READ_WRITE_CACHE,tmp_res.size()*sizeof(float),tmp_res.data());
    }


    void activate(float *layer_res, nn_size_type neuron_count) override
    {
        //get max val
        max_tmp = layer_res[0];
        for (nn_size_type i = 1; i < neuron_count; ++i)
        {
            if (layer_res[i] > max_tmp)
            {
                max_tmp = layer_res[i];
            }
        }

        sum_tmp = 0;
        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            layer_res[i] = expf(layer_res[i] - max_tmp);
            sum_tmp += layer_res[i];
        }

        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            layer_res[i] /= sum_tmp;
        }
    }


    void multiply_neuron_gradient_by_activation_derivative(neuron *neurons, float *layer_res, nn_size_type neuron_count) override
    {
        sum_tmp = 0;//grad_dot_output
        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            sum_tmp += neurons[i].gradient * layer_res[i];
        }

        for (nn_size_type i = 0; i < neuron_count; ++i)
        {
            neurons[i].gradient = layer_res[i] * (neurons[i].gradient - sum_tmp);
        }
    }

    void activate_oclw(std::string layer_res_key, nn_size_type neuron_count)override
    {
        oclw_ptr->process_oclw(km.get("max"), {layer_res_key, tmp_res_key}, {}, {0},neuron_count);

        oclw_ptr->process_oclw(km.get("sum_exp"), {layer_res_key, tmp_res_key}, {}, {0,1},neuron_count);

        oclw_ptr->process_oclw(km.get("activate"), {layer_res_key,tmp_res_key}, {}, {1, neuron_count},neuron_count);
    }


    void multiply_neuron_gradient_by_activation_derivative_oclw(std::string neurons_key,std::string layer_res_key, nn_size_type neuron_count)override
    {
        oclw_ptr->process_oclw(km.get("grad_dot_output"), {neurons_key,layer_res_key, tmp_res_key}, {}, {1},neuron_count);
        oclw_ptr->process_oclw(km.get("multiply_ng_by_activation_derivative"), {neurons_key,layer_res_key, tmp_res_key}, {}, {1},neuron_count);
    }

};

