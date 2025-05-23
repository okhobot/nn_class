#include <layers/Fully_connected_layer.hpp>



Fully_connected_layer::Fully_connected_layer(Activation *a_activation, int a_data_size, int a_neurons_count, float a_weights_dispersion,float a_weights_center, Layer *a_next_layer)
{
    next_layer=a_next_layer;

    data_size=a_data_size;
    neurons_count=a_neurons_count;
    activation=a_activation;

    params_count=data_size*neurons_count;

    weights_dispersion=a_weights_dispersion;
    weights_center=a_weights_center;

}


void Fully_connected_layer::init(int a_layer_index, OCLW *a_oclw)
{
    layer_index=a_layer_index;
    oclw=a_oclw;
    activation->set_oclw(oclw);


    neurons.resize(neurons_count);



    generate_weights(weights_dispersion,weights_center);

    if(oclw->is_inited())
    {
        km.set_default_path("kernels/layers/Fully_connected_layer/");

        km.add_kernel("predict","fcl_lay_predict_kernel");
        km.add_kernel("calculate_gradients_with_ng","fcl_lay_calculate_gradients_with_ng_kernel");
        km.add_kernel("calculate_ng_main_lay","fcl_lay_calculate_ng_with_loss_func_kernel_tmp");
        km.add_kernel("calculate_previous_ng","fcl_lay_calculate_previous_ng_kernel");
        km.add_kernel("calculate_previous_ng_in_neurons","fcl_lay_calculate_previous_ng_in_neurons_kernel");


        weights_key="l_"+std::to_string(layer_index)+"_weights";
        gradients_key="l_"+std::to_string(layer_index)+"_gradients";
        neurons_key="l_"+std::to_string(layer_index)+"_neurons";
        layer_res_key="l_"+std::to_string(layer_index)+"_layer_res";

        oclw->add_and_write_variable(weights_key,CL_READ_WRITE_CACHE,weights.size()*sizeof(nn_type),weights.data());
        oclw->add_variable(gradients_key,CL_READ_WRITE_CACHE,params_count*sizeof(nn_type));

        oclw->add_and_write_variable(neurons_key,CL_READ_WRITE_CACHE,neurons.size()*sizeof(neuron),neurons.data());
        oclw->add_variable(layer_res_key,CL_READ_WRITE_CACHE,neurons_count*sizeof(float));

        weights.clear();
        neurons.clear();
    }
    else
    {
        layer_res.resize(neurons_count);
        gradients.resize(params_count,0);
    }


}




std::vector<float> Fully_connected_layer::predict(std::vector<float> &input)
{

    for(nn_size_type i=0; i<neurons_count; i++)
    {
        layer_res[i]=tof(neurons[i].b);
        for(nn_size_type j=0; j<neurons[i].params_count; j++)
        {
            layer_res[i]+=tof(weights[neurons[i].params_start_index+j])*input[j];
        }
    }



    activation->activate(layer_res.data(),neurons_count);

    return layer_res;
}

std::string Fully_connected_layer::predict_oclw(const std::string &input_key)
{
    oclw->process_oclw(km.get("predict"), {neurons_key,input_key,layer_res_key,weights_key}, {}, {data_size,neurons_count},neurons_count);


    activation->activate_oclw(layer_res_key,neurons_count);
    return layer_res_key;
}


void Fully_connected_layer::calculate_gradients_with_ng(const float *input)
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

void Fully_connected_layer::calculate_gradients_with_ng_oclw(const std::string &input_key)
{
    oclw->process_oclw(km.get("calculate_gradients_with_ng"), {neurons_key,input_key,gradients_key}, {}, {neurons_count, data_size},neurons_count,data_size);
}


void Fully_connected_layer::calculate_ng_main_lay(Loss *loss, const float *input, const float *output)
{
    for(int i=0; i<neurons_count; i++)
    {
        neurons[i].gradient=0;
        loss->add_loss_gradient(output[i],layer_res[i], &neurons[i]);
    }

    activation->multiply_neuron_gradient_by_activation_derivative(neurons.data(), layer_res.data(),neurons_count);

    calculate_gradients_with_ng(input);
}

void Fully_connected_layer::calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)
{
    oclw->process_oclw(km.get("calculate_ng_main_lay"), {neurons_key,output_key,layer_res_key}, {}, {neurons_count},neurons_count);

    activation->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,layer_res_key,neurons_count);

    calculate_gradients_with_ng_oclw(input_key);
}


void Fully_connected_layer::calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)
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
void Fully_connected_layer::calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)
{
    oclw->process_oclw(km.get("calculate_previous_ng_in_neurons"), {previous_neurons_key, neurons_key,weights_key}, {}, {neurons_count, previous_neurons_size},previous_neurons_size);
}
void Fully_connected_layer::calculate_previous_ng(std::vector<float> &previous_gradients)
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
void Fully_connected_layer::calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)
{
    oclw->process_oclw(km.get("calculate_previous_ng"), {neurons_key,weights_key,previous_gradients_key}, {}, {neurons_count, previous_gradients_size},previous_gradients_size);
}

void Fully_connected_layer::calculate_ng(const float *input)
{

    next_layer->calculate_previous_ng_in_neurons(neurons);


    activation->multiply_neuron_gradient_by_activation_derivative(neurons.data(), layer_res.data(),neurons_count);


    calculate_gradients_with_ng(input);
}

void Fully_connected_layer::calculate_ng_oclw(const std::string &input_key)
{

    next_layer->calculate_previous_ng_in_neurons_oclw(neurons_key,neurons_count);

    activation->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,layer_res_key,neurons_count);


    calculate_gradients_with_ng_oclw(input_key);


}

