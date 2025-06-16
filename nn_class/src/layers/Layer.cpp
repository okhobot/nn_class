#include <layers/Layer.hpp>


void Layer::generate_weights(float dispersion, float center)
{
    float d=0;

    dispersion=fromf(dispersion);

    weights.resize(params_count,0);

    if(dispersion<0)dispersion=-dispersion/data_size;


    for(int i=0; i<neurons_count; i++)
    {
        neurons[i].params_count=data_size;
        neurons[i].params_start_index=i*data_size;

        neurons[i].b=fromf(rand()%10001/5000-1);

        neurons[i].b_grad=0;
        neurons[i].gradient=0;

        d=fromf(tof(neurons[i].b)*tof(neurons[i].b)/(data_size+1));

        for(int w_i=0; w_i<data_size; w_i++)
        {
            weights[i*data_size+w_i]=fromf(rand()%10001/5000.0-1) ;
            d+=fromf(tof(weights[i*data_size+w_i])*tof(weights[i*data_size+w_i])/(data_size+1));
        }

        if(dispersion!=0 )
        {
            for(int w_i=0; w_i<data_size; w_i++)
                weights[i*data_size+w_i]=(weights[i*data_size+w_i]*sqrt(dispersion/d)+center);
            neurons[i].b=(neurons[i].b*sqrt(dispersion/d)+center);
        }

    }



}




void Layer::generate_kernels(Loss *loss)
{
    if(!km.have("calculate_ng_main_lay"))return;

    std::string new_kernel_name=km.get("calculate_ng_main_lay").substr(0,km.get("calculate_ng_main_lay").size()-4);
    std::string new_path="kernels/generated/"+new_kernel_name;

    loss->generate_cl_file_with_loss_func(km.get_path("calculate_ng_main_lay"),new_path);

    km.add_kernel("calculate_ng_main_lay",new_kernel_name,"kernels/generated/");
}


void Layer::init_oclw_variables()
{
    weights_key="l_"+std::to_string(layer_index)+"_weights";
    gradients_key="l_"+std::to_string(layer_index)+"_gradients";
    neurons_key="l_"+std::to_string(layer_index)+"_neurons";
    layer_res_key="l_"+std::to_string(layer_index)+"_layer_res";

    oclw_ptr->add_and_write_variable(weights_key,CL_READ_WRITE_CACHE,weights.size()*sizeof(nn_type),weights.data());
    oclw_ptr->add_variable(gradients_key,CL_READ_WRITE_CACHE,params_count*sizeof(nn_type));

    oclw_ptr->add_and_write_variable(neurons_key,CL_READ_WRITE_CACHE,neurons.size()*sizeof(neuron),neurons.data());
    oclw_ptr->add_variable(layer_res_key,CL_READ_WRITE_CACHE,neurons_count*sizeof(float));
}

void Layer::init(int a_layer_index, OCLW *a_oclw_ptr)
{
    layer_index=a_layer_index;
    oclw_ptr=a_oclw_ptr;
    inited=true;

    if(activation_ptr!=0)
        activation_ptr->set_oclw(oclw_ptr);

    neurons.resize(neurons_count);

    generate_weights(weights_dispersion,weights_center);

    if(oclw_ptr->is_inited())
    {
        init_kernels();

        init_oclw_variables();

        weights.clear();
        neurons.clear();
    }
    else
    {
        layer_res.resize(neurons_count,0);
        gradients.resize(params_count,0);
    }
};



void Layer::load(std::ifstream &input)
{
    if(oclw_ptr->is_inited())
    {
        neurons.resize(neurons_count);
        oclw_ptr->read_variable(neurons_key, neurons.size()*sizeof(neuron),neurons.data());
        weights.resize(params_count);
    }
    try
    {
        for(size_t i=0; i<params_count; i++)
            input.read(reinterpret_cast<char*>(&weights[i]),sizeof(nn_type));

        for(int i=0; i<neurons_count; i++)
            input.read(reinterpret_cast<char*>(&neurons[i].b),sizeof(nn_type));
    }
    catch(const char* error_message)
    {
        debug_utils::call_error(0,"Layer::load", "error while loading weights", error_message);
    }
    if(oclw_ptr->is_inited())
    {
        oclw_ptr->write_variable(neurons_key, neurons.size()*sizeof(neuron),neurons.data());
        oclw_ptr->write_variable(weights_key, weights.size()*sizeof(nn_type),weights.data());
        neurons.clear();
        weights.clear();
    }
}

void Layer::save(std::ofstream &output)
{
    if(oclw_ptr->is_inited())
    {
        weights.resize(params_count);
        neurons.resize(neurons_count);
        oclw_ptr->read_variable(neurons_key, neurons.size()*sizeof(neuron),neurons.data());
        oclw_ptr->read_variable(weights_key, weights.size()*sizeof(nn_type),weights.data());
    }

    for(size_t i=0; i<params_count; i++)
        output.write(reinterpret_cast<char*>(&weights[i]),sizeof(nn_type));

    for(int i=0; i<neurons_count; i++)
        output.write(reinterpret_cast<char*>(&neurons[i].b),sizeof(nn_type));

}




std::vector<std::string> Layer::get_kernels_paths()
{
    std::vector<std::string> tmp, res=km.get_kernels_paths();

    if(activation_ptr)tmp=activation_ptr->get_kernels_paths();
    res.insert(res.end(),tmp.begin(),tmp.end());
    return res;
}


void Layer::set_layer_res(const std::vector<float> &res)
{
    if(res.size()>get_layer_res_size())
    {
        debug_utils::call_error(0,"Layer - set_layer_res", "the size of the array is larger than required");
        return;
    }
    layer_res.resize(get_layer_res_size());
    for(int i=0; i<res.size(); i++)layer_res[i]=res[i];

    if(oclw_ptr->is_inited())
    {
        oclw_ptr->write_variable(layer_res_key,layer_res.size()*sizeof(float),layer_res.data());
        layer_res.clear();
    }
}
