#pragma once

#include <layers/Layer.hpp>

class Convolutional_layer: public Layer
{
private:
    std::vector<float> converted_ng, converted_layer_res;
    std::string converted_ng_key,converted_layer_res_key;
    int x_size, y_size, channels_count, nx_size,ny_size, x_step,y_step,  lay_res_x_size,lay_res_y_size, neuron_out_size;
    nn_size_type out_size, input_size;

protected:

    void calculate_gradients_with_ng(const float *input) override;
    void calculate_gradients_with_ng_oclw(const std::string &input_key) override;

public:
    Convolutional_layer(Activation *a_activation,
                        int a_x_size, int a_y_size, int a_channels_count, int a_neurons_count, int a_nx_size=3, int a_ny_size=3, int a_x_step=1, int a_y_step=1,
                        float a_weights_dispersion=-1,float a_weights_center=0, Layer *a_next_layer_ptr=0);
    // a_activation - layer activation
    // a_x_size, a_y_size - input image size
    // a_channels_count - channels_count; for rgb image a_channels_count=3
    // a_neurons_count - filters count
    // a_nx_size, a_ny_size - filter size
    // a_x_step, a_y_step - filter steps
    // a_weights_center - center of weights_dispersion
    // a_next_layer - next layer ptr; if (a_next_layer_ptr=0) it set when init calls


    void init(int a_layer_index, OCLW *a_oclw_ptr) override;

    std::vector<float> predict(std::vector<float> &input) override;
    std::string predict_oclw(const std::string &input_key) override;//returns layer_res_key

    void calculate_ng_main_lay(Loss *loss, const float *input, const float *output) override;
    void calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)override;


    void calculate_ng(const float *input) override;
    void calculate_ng_oclw(const std::string &input_key) override;


    void calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)override;
    void calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)override;
    void calculate_previous_ng(std::vector<float> &previous_gradients)override;
    void calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)override;


    nn_size_type get_input_size() override
    {
        return input_size;
    }

    nn_size_type get_layer_res_size() override
    {
        return out_size;
    }

};


