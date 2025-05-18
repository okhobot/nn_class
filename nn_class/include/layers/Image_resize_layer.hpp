#pragma once

#include <layers/Layer.hpp>

class Image_resize_layer : public Layer
{
protected:
    std::vector<float> next_gradients;
    std::string next_gradients_key;

    int old_x_size,old_y_size,new_x_size,new_y_size,channels_count,frame_size, out_size;
    float resize_coef_x, resize_coef_y, data_normalization_coef,
          inversed_resize_coef_x, inversed_resize_coef_y, inversed_data_normalization_coef;

    inline void resize_image(std::vector<float> &img_from, std::vector<float> &img_to, float rc_x, float rc_y, float dnc, int n_size_x,int n_size_y, int frame_s, int o_size_x,int o_size_y);


public:
    Image_resize_layer(int a_old_x_size, int a_old_y_size, int a_new_x_size, int a_new_y_size, int a_channels_count=1, int a_frame_size=0, Layer *a_next_layer=0);
    // a_old_x_size, a_old_y_size - image start dimensions
    // a_new_x_size, a_new_y_size - image out dimensions
    // a_channels_count - channels_count; for rgb channels_count=3
    // a_frame_size - the number of pixels that will be filled with 0 on each side of the image
    // a_next_layer - next layer ptr; if (a_next_layer==0) it set when init calls

    void init(int a_layer_index, OCLW *a_oclw) override;

    std::vector<float> predict(std::vector<float> &input)override;
    std::string predict_oclw(const std::string &input_key)override;

    void calculate_ng_main_lay(Loss *loss, const float *input, const float *output) override;
    void calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)override;

    void calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)override;
    void calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)override;
    void calculate_previous_ng(std::vector<float> &previous_gradients)override;
    void calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)override;


    void load(std::ifstream &input)override {};

    void save(std::ofstream &output)override {};

    nn_size_type get_layer_res_size() override
    {
        return out_size;
    }
};
