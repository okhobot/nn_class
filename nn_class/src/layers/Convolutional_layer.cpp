#include <layers/Convolutional_layer.hpp>

Convolutional_layer::Convolutional_layer(Activation *a_activation, int a_x_size, int a_y_size, int a_channels_count, int a_neurons_count, int a_nx_size, int a_ny_size, int a_x_step, int a_y_step, float a_weights_dispersion,float a_weights_center, Layer *a_next_layer)
{
    next_layer=a_next_layer;

    nx_size=a_nx_size;
    ny_size=a_ny_size;
    x_step=a_x_step;
    y_step=a_y_step;
    x_size=a_x_size;
    y_size=a_y_size;

    channels_count=a_channels_count;

    lay_res_x_size=((x_size-nx_size)/x_step+1);
    lay_res_y_size=((y_size-ny_size)/y_step+1);

    input_size=channels_count*x_size*y_size;
    data_size=nx_size*ny_size*channels_count;
    neurons_count=a_neurons_count;
    activation=a_activation;

    out_size=neurons_count*((x_size-nx_size)/x_step+1)*((y_size-ny_size)/y_step+1);
    neuron_out_size=out_size/neurons_count;

    params_count=nx_size*ny_size*channels_count*neurons_count;


    weights_dispersion=a_weights_dispersion;
    weights_center=a_weights_center;




}

void Convolutional_layer::init(int a_layer_index, OCLW *a_oclw)
{
    layer_index=a_layer_index;
    oclw=a_oclw;
    activation->set_oclw(oclw);


    neurons.resize(neurons_count);

    generate_weights(weights_dispersion,weights_center);

    if(oclw->is_inited())
    {

        km.set_default_path("kernels/layers/Convolutional_layer/");

        km.add_kernel("predict","conv_lay_predict_kernel");
        km.add_kernel("calculate_gradients_with_ng","conv_lay_calculate_gradients_with_ng_kernel");
        km.add_kernel("calculate_ng_main_lay","conv_lay_calculate_ng_with_loss_func_kernel_tmp");
        km.add_kernel("calculate_previous_ng","conv_lay_calculate_previous_ng_kernel");
        km.add_kernel("calculate_previous_ng_in_neurons","conv_lay_calculate_previous_ng_in_neurons_kernel");

        km.add_kernel("convert_to_neurons","conv_lay_convert_to_neurons_kernel");


        weights_key="l_"+std::to_string(layer_index)+"_weights";
        gradients_key="l_"+std::to_string(layer_index)+"_gradients";
        neurons_key="l_"+std::to_string(layer_index)+"_neurons";
        layer_res_key="l_"+std::to_string(layer_index)+"_layer_res";
        converted_ng_key="l_"+std::to_string(layer_index)+"_converted_ng";
        converted_layer_res_key="l_"+std::to_string(layer_index)+"_converted_layer_res";

        oclw->add_and_write_variable(weights_key,CL_READ_WRITE_CACHE,weights.size()*sizeof(nn_type),weights.data());
        oclw->add_variable(gradients_key,CL_READ_WRITE_CACHE,params_count*sizeof(nn_type));

        oclw->add_and_write_variable(neurons_key,CL_READ_WRITE_CACHE,neurons.size()*sizeof(neuron),neurons.data());
        oclw->add_variable(layer_res_key,CL_READ_WRITE_CACHE,out_size*sizeof(float));

        oclw->add_variable(converted_ng_key,CL_READ_WRITE_CACHE,out_size*sizeof(float));
        oclw->add_variable(converted_layer_res_key,CL_READ_WRITE_CACHE,neurons_count*sizeof(float));

        weights.clear();
        neurons.clear();
    }
    else
    {
        layer_res.resize(out_size);
        converted_ng.resize(out_size);
        converted_layer_res.resize(neurons_count);
        gradients.resize(params_count,0);
    }

    inited=true;
}

std::vector<float> Convolutional_layer::predict(std::vector<float> &input)
{


    int res_index=0;
    for(int neuron_index=0; neuron_index<neurons_count; neuron_index++)
        for(int y=0; y+ny_size<=y_size; y+=y_step)
            for(int x=0; x+nx_size<=x_size; x+=x_step)
            {
                layer_res[res_index]=0;

                for(int channel=0; channel<channels_count; channel++)
                    for(int j=0; j<ny_size; j++)
                        for(int k=0; k<nx_size; k++)
                        {
                            layer_res[res_index]+=tof(weights[neurons[neuron_index].params_start_index+ k+j*nx_size+channel*nx_size*ny_size])*
                                                  input[x+k+(y+j)*x_size+channel*x_size*y_size];
                        }

                ++res_index;
            }


    activation->activate(layer_res.data(),layer_res.size());

    //for(int i=0;i<layer_res.size();i++)std::cout<<layer_res[i]<<" ";
    //std::cout<<" li:"<<layer_index<<std::endl;

    return layer_res;
}

std::string Convolutional_layer::predict_oclw(const std::string &input_key)
{
    oclw->process_oclw(km.get("predict"), {neurons_key,input_key,layer_res_key,weights_key}, {},
    {channels_count, nx_size,ny_size, lay_res_x_size,lay_res_y_size,x_step,y_step,  neurons_count, y_size, x_size},neurons_count, y_size/y_step, x_size/x_step);


    activation->activate_oclw(layer_res_key,out_size);
    return layer_res_key;
}

void Convolutional_layer::calculate_gradients_with_ng(const float *input)
{
    //std::cout<<neurons[0].gradient<<" "<<layer_index<<std::endl;

    for(int channel=0; channel<channels_count; channel++)
        for(int j=0; j<ny_size; j++)
            for(int k=0; k<nx_size; k++)
                for(int neuron_index=0; neuron_index<neurons_count; neuron_index++)
                    for(int y=0; y+ny_size<=y_size; y+=y_step)
                        for(int x=0; x+nx_size<=x_size; x+=x_step)
                        {

                            add(&gradients[neurons[neuron_index].params_start_index+ k+j*nx_size+channel*nx_size*ny_size],
                                input[x+k+(y+j)*x_size+channel*x_size*y_size]
                                *fromf(neurons[neuron_index].gradient) );
                        }

}
void Convolutional_layer::calculate_gradients_with_ng_oclw(const std::string &input_key)
{
    //oclw->process_oclw(km.get("calculate_gradients_with_ng"), {neurons_key,input_key,gradients_key}, {}, {neurons_count, data_size},neurons_count,data_size);

    oclw->process_oclw(km.get("calculate_gradients_with_ng"), {neurons_key,input_key,gradients_key}, {},
    {neurons_count, x_size, y_size, lay_res_x_size,lay_res_y_size,x_step,y_step,  channels_count, ny_size,nx_size},channels_count, ny_size, nx_size);
}

void Convolutional_layer::calculate_ng_main_lay(Loss *loss, const float *input, const float *output)
{

    for(int i=0; i<neurons_count; i++)
    {
        neurons[i].gradient=0;
        converted_layer_res[i]=0;
        for(int j=i*neuron_out_size; j<(i+1)*neuron_out_size; j++)
        {
            loss->add_loss_gradient(output[j],layer_res[j], &neurons[i]);
            converted_layer_res[i]+=layer_res[j];
        }
        neurons[i].gradient/=neuron_out_size;
        converted_layer_res[i]/=neuron_out_size;

    }

    activation->multiply_neuron_gradient_by_activation_derivative(neurons.data(), converted_layer_res.data(),neurons_count);

    calculate_gradients_with_ng(input);
}


void Convolutional_layer::calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)
{
    oclw->process_oclw(km.get("calculate_ng_main_lay"), {neurons_key,output_key,layer_res_key, converted_layer_res_key}, {}, {neuron_out_size,neurons_count},neurons_count);

    activation->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,converted_layer_res_key,neurons_count);

    calculate_gradients_with_ng_oclw(input_key);
}


void Convolutional_layer::calculate_ng(const float *input)
{
    next_layer->calculate_previous_ng(converted_ng);



    for(int i=0; i<neurons_count; i++)
    {
        neurons[i].gradient=0;
        converted_layer_res[i]=0;
        for(int j=i*neuron_out_size; j<(i+1)*neuron_out_size; j++)
        {
            neurons[i].gradient+=converted_ng[j];
            converted_layer_res[i]+=layer_res[j];
        }
        neurons[i].gradient/=neuron_out_size;
        converted_layer_res[i]/=neuron_out_size;

    }


    activation->multiply_neuron_gradient_by_activation_derivative(neurons.data(), converted_layer_res.data(),neurons_count);


    calculate_gradients_with_ng(input);
}

void Convolutional_layer::calculate_ng_oclw(const std::string &input_key)
{

    next_layer->calculate_previous_ng_oclw(converted_ng_key,out_size);

    oclw->process_oclw(km.get("convert_to_neurons"), {neurons_key, converted_ng_key, layer_res_key, converted_layer_res_key}, {}, {neuron_out_size, neurons_count},neurons_count);

    activation->multiply_neuron_gradient_by_activation_derivative_oclw(neurons_key,converted_layer_res_key,neurons_count);


    calculate_gradients_with_ng_oclw(input_key);


}



void Convolutional_layer::calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)
{
    for(int i=0; i<previous_neurons.size(); i++)
    {
        previous_neurons[i].gradient=0;
        for(int j=0; j<neurons_count; j++)
        {
            for(int x=0; x<fmin(fmin(x_size-i%x_size, i%x_size+1),nx_size); x++)
                for(int y=0; y<fmin(fmin(y_size-i/x_size, i/x_size+1),ny_size); y++)
                {
                    previous_neurons[i].gradient+=
                        neurons[j].gradient*
                        tof(weights[neurons[j].params_start_index+x+y*nx_size+i/(x_size*y_size)*nx_size*ny_size]);
                }
        }
    }
}
void Convolutional_layer::calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)
{
    oclw->process_oclw(km.get("calculate_previous_ng_in_neurons"), {previous_neurons_key,neurons_key,weights_key}, {}, {neurons_count,nx_size,ny_size,x_size,y_size, previous_neurons_size},previous_neurons_size);
}

void Convolutional_layer::calculate_previous_ng(std::vector<float> &previous_gradients)
{

    for(int i=0; i<previous_gradients.size(); i++)
    {
        previous_gradients[i]=0;
        for(int j=0; j<neurons_count; j++)
        {
            for(int x=0; x<fmin(fmin(x_size-i%x_size, i%x_size+1),nx_size); x++)
                for(int y=0; y<fmin(fmin(y_size-i/x_size, i/x_size+1),ny_size); y++)
                {
                    previous_gradients[i]+=
                        neurons[j].gradient*
                        tof(weights[neurons[j].params_start_index+x+y*nx_size+i/(x_size*y_size)*nx_size*ny_size]);
                }
        }
    }
}
void Convolutional_layer::calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)
{
    oclw->process_oclw(km.get("calculate_previous_ng"), {neurons_key,weights_key,previous_gradients_key}, {}, {neurons_count,nx_size,ny_size,x_size,y_size, previous_gradients_size},previous_gradients_size);
}




