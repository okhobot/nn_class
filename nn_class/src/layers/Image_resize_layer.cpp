#include <layers/Image_resize_layer.hpp>


void Image_resize_layer::resize_image(std::vector<float> &img_from, std::vector<float> &img_to, float rc_x, float rc_y, float dnc, int n_size_x,int n_size_y, int frame_s, int o_size_x,int o_size_y)
{
    float res_x, res_y;
    for(int channel=0; channel<channels_count; ++channel)
    {
        res_y=frame_s;
        for(int y=0; y<o_size_y; ++y)
        {
            res_x=frame_s;
            for(int x=0; x<o_size_x; ++x)
            {
                img_to[channel*n_size_x*n_size_y+((int)res_y)*n_size_x+ ((int)res_x)]+=img_from[channel*o_size_x*old_y_size+y*o_size_x+x]*dnc;

                if(rc_x>1||rc_y>1)
                {
                    float fill_e=img_to[channel*n_size_x*n_size_y+((int)res_y)*n_size_x+ ((int)res_x)];

                    for(int dy=(int)res_y; dy<(int)(res_y+rc_y); ++dy)
                        for(int dx=(int)res_x; dx<(int)(res_x+rc_x); ++dx)
                            img_to[channel*n_size_x*n_size_y+dy*n_size_x+ dx]=fill_e;

                }
                res_x+=rc_x;
            }
            res_y+=rc_y;
        }
    }
}





Image_resize_layer::Image_resize_layer(int a_old_x_size, int a_old_y_size, int a_new_x_size, int a_new_y_size, int a_channels_count, int a_frame_size, Layer *a_next_layer)
{
    frame_size=a_frame_size;
    old_x_size=a_old_x_size;
    old_y_size=a_old_y_size;
    new_x_size=a_new_x_size+2*frame_size;
    new_y_size=a_new_y_size+2*frame_size;
    channels_count=a_channels_count;

    activation=0;
    neurons_count=0;
    params_count=0;
    data_size=channels_count*old_x_size*old_y_size;
    out_size=channels_count*new_x_size*new_y_size;

    next_layer=a_next_layer;

    resize_coef_x=(float)(new_x_size-2*frame_size)/old_x_size;
    resize_coef_y=(float)(new_y_size-2*frame_size)/old_y_size;
    data_normalization_coef=(resize_coef_x>=1?1:resize_coef_x)*(resize_coef_y>=1?1:resize_coef_y);

    inversed_resize_coef_x=old_x_size/(float)new_x_size;
    inversed_resize_coef_y=old_y_size/(float)new_y_size;
    inversed_data_normalization_coef=(1/resize_coef_x>=1?1:1/resize_coef_x)*(1/resize_coef_y>=1?1:1/resize_coef_y);

    //std::cout<<inversed_data_normalization_coef<<" "<<1/resize_coef_x<<" "<<1/resize_coef_y<<std::endl;
}


void Image_resize_layer::init(int a_layer_index, OCLW *a_oclw)
{
    layer_index=a_layer_index;
    oclw=a_oclw;



    generate_weights(weights_dispersion,weights_center);

    if(oclw->is_inited())
    {
        km.set_default_path("kernels/layers/Image_resize_layer/");

        km.add_kernel("fill_floats","fill_float_arr_kernel");
        km.add_kernel("fill_ng","fill_neurons_gradients_kernel");
        km.add_kernel("resize","imres_lay_resize_kernel");
        km.add_kernel("resize_in_neurons","imres_lay_resize_in_neurons_kernel");

        km.add_kernel("calculate_ng_main_lay","imres_lay_calculate_ng_with_loss_func_kernel_tmp");


        weights_key=oclw_null;
        gradients_key=oclw_null;
        neurons_key=oclw_null;


        layer_res_key="l_"+std::to_string(layer_index)+"_layer_res";
        next_gradients_key="l_"+std::to_string(layer_index)+"_next_gradients";


        oclw->add_variable(layer_res_key,CL_READ_WRITE_CACHE,out_size*sizeof(float));
        oclw->add_variable(next_gradients_key,CL_READ_WRITE_CACHE,out_size*sizeof(float));

    }
    else
    {
        layer_res.resize(out_size);
        next_gradients.resize(out_size);

        if(layer_index==0)neurons.resize(1);
    }

    inited=true;
}



std::vector<float> Image_resize_layer::predict(std::vector<float> &input)
{
    std::fill(layer_res.begin(),layer_res.end(),0);
    resize_image(input,layer_res,resize_coef_x, resize_coef_y, data_normalization_coef, new_x_size,new_y_size, frame_size,old_x_size,old_y_size);

    //for(int i=0;i<layer_res.size();i++)std::cout<<layer_res[i]<<" ";
    //std::cout<<" li:"<<layer_index<<std::endl;

    return layer_res;
}

std::string Image_resize_layer::predict_oclw(const std::string &input_key)
{
    oclw->process_oclw(km.get("fill_floats"), {layer_res_key}, {0}, {out_size},out_size);
    oclw->process_oclw(km.get("resize"), {input_key, layer_res_key}, {resize_coef_x,resize_coef_y, data_normalization_coef},
    {new_x_size,new_y_size, frame_size,  channels_count, old_y_size, old_x_size},channels_count, old_y_size, old_x_size);


    return layer_res_key;
}

void Image_resize_layer::calculate_previous_ng(std::vector<float> &previous_gradients)
{
    std::fill(previous_gradients.begin(),previous_gradients.end(),0);

    if(layer_index>0)
        next_layer->calculate_previous_ng(next_gradients);

    resize_image(next_gradients,previous_gradients, inversed_resize_coef_x, inversed_resize_coef_y, inversed_data_normalization_coef, old_x_size,old_y_size,0,new_x_size,new_y_size);

///resize_image(std::vector<float> &img_from, std::vector<float> &img_to, float rc_x, float rc_y, float dnc, int n_size_x,int n_size_y, int frame_s, int o_size_x,int o_size_y);
    //if(layer_index==2)std::cout<<previous_gradients.size()<<" | "<<old_x_size<<std::endl;
    //for(int i=0;i<previous_neurons.size();i++)previous_neurons[i].gradient=0;
}

void Image_resize_layer::calculate_previous_ng_oclw(const std::string &previous_gradients_key, size_t previous_gradients_size)
{
    oclw->process_oclw(km.get("fill_floats"), {previous_gradients_key}, {0}, {previous_gradients_size},previous_gradients_size);

    if(layer_index>0)
        next_layer->calculate_previous_ng_oclw(next_gradients_key,out_size);

    oclw->process_oclw(km.get("resize"), {next_gradients_key, previous_gradients_key}, {inversed_resize_coef_x, inversed_resize_coef_y, inversed_data_normalization_coef},
    {old_x_size,old_y_size, 0,  channels_count, new_y_size, new_x_size},channels_count, new_y_size, new_x_size);
}


void Image_resize_layer::calculate_previous_ng_in_neurons(std::vector<neuron> &previous_neurons)
{

    for(int i=0; i<previous_neurons.size(); i++)previous_neurons[i].gradient=0;

    if(layer_index>0)
        next_layer->calculate_previous_ng(next_gradients);



    float res_x, res_y;
    for(int channel=0; channel<channels_count; ++channel)
    {
        res_y=0;
        for(int y=0; y<new_y_size; ++y)
        {
            res_x=0;
            for(int x=0; x<new_x_size; ++x)
            {

                previous_neurons[channel*old_x_size*old_y_size+((int)res_y)*old_x_size+ ((int)res_x)].gradient+=next_gradients[channel*new_x_size*old_y_size+y*new_x_size+x]*inversed_data_normalization_coef;

                if(inversed_resize_coef_x>1||inversed_resize_coef_y>1)
                {
                    float fill_e=previous_neurons[channel*old_x_size*old_y_size+((int)res_y)*old_x_size+ ((int)res_x)].gradient;

                    for(int dy=(int)res_y; dy<(int)(res_y+inversed_resize_coef_y); ++dy)
                        for(int dx=(int)res_x; dx<(int)(res_x+inversed_resize_coef_x); ++dx)
                            previous_neurons[channel*old_x_size*old_y_size+dy*old_x_size+ dx].gradient=fill_e;

                }
                res_x+=inversed_resize_coef_x;
            }
            res_y+=inversed_resize_coef_y;
        }
    }
}
void Image_resize_layer::calculate_previous_ng_in_neurons_oclw(const std::string &previous_neurons_key, size_t previous_neurons_size)
{
    oclw->process_oclw(km.get("fill_ng"), {previous_neurons_key}, {0}, {previous_neurons_size},previous_neurons_size);

    if(layer_index>0)
        next_layer->calculate_previous_ng_oclw(next_gradients_key,out_size);

    oclw->process_oclw(km.get("resize_in_neurons"), {next_gradients_key, previous_neurons_key}, {inversed_resize_coef_x, inversed_resize_coef_y, inversed_data_normalization_coef},
    {old_x_size,old_y_size, 0,  channels_count, new_y_size, new_x_size},channels_count, new_y_size, new_x_size);
}



void Image_resize_layer::calculate_ng_main_lay(Loss *loss, const float *input, const float *output)
{

    for(int i=0; i<out_size; i++)
    {
        neurons[0].gradient=0;
        loss->add_loss_gradient(output[i],layer_res[i], &neurons[0]);
        next_gradients[i]=neurons[0].gradient;
    }
}

void Image_resize_layer::calculate_ng_main_lay_oclw(const std::string &input_key, const std::string &output_key)
{
    oclw->process_oclw(km.get("calculate_ng_main_lay"), {output_key,layer_res_key, next_gradients_key}, {}, {out_size},out_size);
}
