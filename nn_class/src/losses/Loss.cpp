#include <losses/Loss.hpp>



void Loss::generate_cl_file_with_loss_func(std::string tmp_file_path, std::string file_to_save_path)
{
    std::string data;
    std::stringstream ss;
    std::ifstream inp(tmp_file_path+".cl");
    ss<<inp.rdbuf();
    inp.close();


    data=ss.str();
    ss.clear();

    int it=data.find(com_to_call);
    while(it!=-1)
    {
        data.replace(data.begin()+it,data.begin()+it+com_to_call.size(),inline_add_loss_gradient);
        it=data.find(com_to_call);
    }

    std::ofstream outp(file_to_save_path+".cl");
    outp<<data;
    outp.close();

}

float Loss::calculate_error(float *output, float *layer_res, const nn_size_type &output_size)
{
    float res=0;
    for(nn_size_type i=0; i<output_size; i++)
        res+=(output[i]-layer_res[i])*(output[i]-layer_res[i]);
    return res;
}

void Loss::add_loss_gradient(const float &output, const float &layer_res, neuron *a_neuron)
{
    a_neuron->gradient+=2*(output-layer_res);
}
