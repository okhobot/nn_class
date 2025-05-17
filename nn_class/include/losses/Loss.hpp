#pragma once
#include <neuron.h>

#include <iostream>
#include <fstream>
#include <sstream>

class Loss ///Default - MSE
{
protected:
    std::string inline_add_loss_gradient="2*(output[i]-layer_res[i])";
    std::string com_to_call="loss_gradient()";


public:

    virtual void generate_cl_file_with_loss_func(std::string tmp_file_path, std::string file_to_save_path); // generate kernel with inline_add_loss_gradient instead of com_to_call

    virtual float calculate_error(float *output, float *layer_res, const nn_size_type &output_size);// calculate error for logs

    virtual void add_loss_gradient(const float &output, const float &layer_res, neuron *neuron);// calculate neuron gradient in the main layer
};
