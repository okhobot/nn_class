#include <layers/Dense_layer.hpp>



Dense_layer::Dense_layer(Activation *a_activation, int a_data_size, int a_neurons_count, float a_weights_dispersion,float a_weights_center, Layer *a_next_layer)
{
    next_layer_ptr=a_next_layer;

    data_size=a_data_size;
    neurons_count=a_neurons_count;
    activation_ptr=a_activation;

    params_count=data_size*neurons_count;

    weights_dispersion=a_weights_dispersion;
    weights_center=a_weights_center;

}


void Dense_layer::init_kernels()
{
    km.set_default_path("kernels/layers/Dense_layer/");

    km.add_kernel("predict","dense_lay_predict_kernel");
    km.add_kernel("calculate_gradients_with_ng","dense_lay_calculate_gradients_with_ng_kernel");
    km.add_kernel("calculate_ng_main_lay","dense_lay_calculate_ng_with_loss_func_kernel_tmp");
    km.add_kernel("calculate_previous_ng","dense_lay_calculate_previous_ng_kernel");
    km.add_kernel("calculate_previous_ng_in_neurons","dense_lay_calculate_previous_ng_in_neurons_kernel");
}




std::vector<float> Dense_layer::predict(std::vector<float> &input)
{

    for(nn_size_type i=0; i<neurons_count; i++)
    {
        layer_res[i]=tof(neurons[i].b);
        for(nn_size_type j=0; j<neurons[i].params_count; j++)
        {
            layer_res[i]+=tof(weights[neurons[i].params_start_index+j])*input[j];
        }
    }



    activation_ptr->activate(layer_res.data(),neurons_count);

    return layer_res;
}

std::string Dense_layer::predict_oclw(const std::string &input_key)
{
    oclw_ptr->process_oclw(km.get("predict"), {neurons_key,input_key,layer_res_key,weights_key}, {}, {data_size,neurons_count},neurons_count);


    activation_ptr->activate_oclw(layer_res_key,neurons_count);
    return layer_res_key;
}


void Dense_layer::calculate_gradients_with_ng(const float *input)
{
    for(int n_i=0; n_i<neurons_count; n_i++)
    {
        for(int inp_i=0; inp_i<data_size; inp_i++)
        {
            add(&gradients[n_i*neurons[n_i].params_count+inp_i],fromf(input[inp_i]*neurons[n_i].gradient));
        }
        neurons[n_i].b_grad+=fromf(neurons[n_i].gradient);
    }
}

void Dense_layer::calculate_gradients_with_ng_oclw(const std::string &input_key)
{
    oclw_ptr->process_oclw(km.get("calculate_gradients_with_ng"), {neurons_key,input_key,gradients_key}, {}, {neurons_count, data_size},neurons_count,data_size);
}


void Dense_layer::calculate_ng_main_lay(Loss *loss, const float *input, const float *output)
{
    for(int i=0; i<neurons_count; i++)
    {
        neurons[i].gradient=0;
        loss->add_loss_gradient(output[i],layer_res[i], &neurons[i]);
    }

    activation_ptr->multiply_neuron_gradient_by_activation_derivative(neurons.data(), layer_res.data(),neurons_count);

    calculate_gradients_with_ng(input);
}

void Dense_layer::calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)
{
    oclw_ptr->process_oclw(km.get("calculate_ng_main_lay"), {neurons_key,output_key,layer_res_key}, {}, {neurons_count},neurons_count);

    activation_ptr->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,layer_res_key,neurons_count);

    calculate_gradients_with_ng_oclw(input_key);
}


void Dense_layer::calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)
{
    for(int i=0; i<previous_neurons.size(); i++)
    {
        previous_neurons[i].gradient=0;
        for(int j=0; j<neurons_count; j++)
        {
            previous_neurons[i].gradient+=
                neurons[j].gradient*
                tof(weights[neurons[j].params_start_index+i]);
        }
    }
}
void Dense_layer::calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)
{
    oclw_ptr->process_oclw(km.get("calculate_previous_ng_in_neurons"), {previous_neurons_key, neurons_key,weights_key}, {}, {neurons_count, previous_neurons_size},previous_neurons_size);
}
void Dense_layer::calculate_previous_ng(std::vector<float> &previous_gradients)
{
    for(int i=0; i<previous_gradients.size(); i++)
    {
        previous_gradients[i]=0;
        for(int j=0; j<neurons_count; j++)
        {
            previous_gradients[i]+=
                neurons[j].gradient*
                tof(weights[neurons[j].params_start_index+i]);
        }
    }
}
void Dense_layer::calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)
{
    oclw_ptr->process_oclw(km.get("calculate_previous_ng"), {neurons_key,weights_key,previous_gradients_key}, {}, {neurons_count, previous_gradients_size},previous_gradients_size);
}

void Dense_layer::calculate_ng(const float *input)
{

    next_layer_ptr->calculate_previous_ng_in_neurons(neurons);


    activation_ptr->multiply_neuron_gradient_by_activation_derivative(neurons.data(), layer_res.data(),neurons_count);


    calculate_gradients_with_ng(input);
}

void Dense_layer::calculate_ng_oclw(const std::string &input_key)
{

    next_layer_ptr->calculate_previous_ng_in_neurons_oclw(neurons_key,neurons_count);

    activation_ptr->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,layer_res_key,neurons_count);


    calculate_gradients_with_ng_oclw(input_key);


}

