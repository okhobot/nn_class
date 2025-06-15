#pragma once

#include <neuron.h>
#include <oclw.hpp>
#include <Kernels_manager.hpp>
#include <activations/Activation.hpp>
#include <losses/Loss.hpp>

#include <vector>
#include <cmath>

class Layer
{

protected:
    Kernels_manager km;

    std::vector<neuron> neurons;

    std::vector<nn_type> weights, gradients;
    std::vector<float> layer_res;

    std::string weights_key, gradients_key, neurons_key, layer_res_key;

    std::string path_to_kernels;

    Layer *next_layer_ptr=0;
    Activation *activation_ptr=0;
    OCLW *oclw_ptr=0;

    float weights_dispersion=0,weights_center=0;
    nn_size_type params_count=0;
    nn_size_type data_size=0, neurons_count=0;
    int layer_index=0;

    bool inited=false;



    virtual void generate_weights(float dispersion, float center);

    virtual void calculate_gradients_with_ng(const float *input) {};
    virtual void calculate_gradients_with_ng_oclw(const std::string &input_key) {};


public:

    virtual void init(int a_layer_index, OCLW *a_oclw_ptr)
    {
        layer_index=a_layer_index;
        oclw_ptr=a_oclw_ptr;
        inited=true;
    };

    virtual std::vector<float> predict(std::vector<float> &input)
    {
        return layer_res;
    };
    virtual std::string predict_oclw(const std::string &input_key)// input_key - key of buffer in oclw dict
    {
        return layer_res_key;
    };

    virtual void calculate_ng_main_lay(Loss *loss,const float *input, const float *output) {};
    virtual void calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key) {};

    //functions for back propagation
    virtual void calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons) {};
    virtual void calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size) {};
    virtual void calculate_previous_ng(std::vector<float> &previous_gradients) {};
    virtual void calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size) {};

    virtual void calculate_ng(const float *input) {};
    virtual void calculate_ng_oclw(const std::string &input_key) {};
    //

    virtual void load(std::ifstream &input);

    virtual void save(std::ofstream &output);

    virtual void generate_kernels(Loss *loss);// if need to generate some kernels

    /// getters

    virtual std::vector<std::string> get_kernels_paths();

    virtual std::string get_neurons_key()
    {
        return neurons_key;
    }

    virtual std::string get_weights_key()
    {
        return weights_key;
    }

    virtual std::string get_gradients_key()
    {
        return gradients_key;
    }

    virtual std::string get_layer_res_key()
    {
        return layer_res_key;
    }


    virtual float* get_layer_res_ptr()
    {
        return layer_res.data();
    }

    virtual nn_type* get_weights_ptr()
    {
        return weights.data();
    }

    virtual nn_type* get_gradients_ptr()
    {
        return gradients.data();
    }


    virtual neuron* get_neurons_ptr()
    {
        return neurons.data();
    }


    virtual nn_size_type get_params_count()
    {
        return params_count;
    }

    virtual nn_size_type get_input_size()
    {
        return data_size;
    }

    virtual nn_size_type get_neurons_count()
    {
        return neurons_count;
    }

    virtual nn_size_type get_layer_res_size()
    {
        return neurons_count;
    }

    virtual nn_size_type get_layer_index()
    {
        return layer_index;
    }

    virtual size_t get_layer_data_size()
    {
        return params_count+neurons_count;
    }

    ///setters
    virtual void set_next_layer(Layer *a_next_layer, bool forced=false)
    {
        if(next_layer_ptr==0 || forced)
            next_layer_ptr=a_next_layer;
    }

    virtual void set_layer_res(const std::vector<float> &res);

    virtual bool is_inited()
    {
        return inited;
    }
};
