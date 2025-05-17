#pragma once

#include <layers/Layer.hpp>


class Fully_connected_layer : public Layer
{



    void calculate_gradients_with_ng(const float *input)override;
    void calculate_gradients_with_ng_oclw(const std::string &input_key)override;


public:
    Fully_connected_layer() {};
    Fully_connected_layer(Activation *a_activation, int a_data_size, int a_neurons_count, float a_weights_dispersion=-1,float a_weights_center=0, Layer *a_next_layer=0);
    // a_activation - layer activation
    // a_data_size - input size
    // a_neurons_count - out size
    // a_weights_center - center of weights_dispersion
    // a_next_layer - next layer ptr; if (a_next_layer==0) it set when init calls

    void init(int a_layer_index, OCLW *a_oclw)override;

    std::vector<float> predict(std::vector<float> &input)override;
    std::string predict_oclw(const std::string &input_key)override;//returns layer_res_key

    void calculate_ng_main_lay(Loss *loss, const float *input, const float *output)override;
    void calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)override;

    void calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)override;
    void calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)override;
    void calculate_previous_ng(std::vector<float> &previous_gradients)override;
    void calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)override;

    void calculate_ng(const float *input) override;
    void calculate_ng_oclw(const std::string &input_key) override;

};
