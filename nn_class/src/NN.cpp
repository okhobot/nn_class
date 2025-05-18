#include <NN.hpp>




NN::NN(Optimizer *a_optimizer, Loss *a_loss)
{
    optimizer=a_optimizer;
    loss=a_loss;
}


void NN::show()
{
    if(logs_output==0)return;
    nn_size_type params_count=0;
    for(int i=layers.size()-1; i>=0; i--)
    {
        (*logs_output)<<"layer index: "<<layers[i]->get_layer_index()<<
                      "; type: "<<typeid(*layers[i]).name()<<
                      "; params count: "<<layers[i]->get_params_count()<<
                      "; input size: "<<layers[i]->get_input_size()<<
                      "; output size: "<<layers[i]->get_layer_res_size()<<
                      std::endl;

        params_count+=layers[i]->get_params_count();
    }
    (*logs_output)<<"model total params count: "<<params_count<<std::endl;

}



void NN::init(int device_index, unsigned seed)
{
    std::vector<std::string> kernels, tmp;

    if(device_index>-1) oclw.init(device_index);
    optimizer->set_oclw(&oclw);

    tmp=optimizer->get_kernels_paths();
    kernels.insert(kernels.end(),tmp.begin(), tmp.end());

    srand(seed);
    for(int i=0; i<layers.size(); i++)
    {
        layers[i]->init(i,&oclw);
        if(i>0)layers[i]->set_next_layer(layers[i-1]);

        if(device_index>-1)
        {
            oclw.add_variable(nn_input_key,CL_READ_ONLY_CACHE,layers[layers.size()-1]->get_input_size()*sizeof(float));
            oclw.add_variable(nn_output_key,CL_READ_ONLY_CACHE,layers[0]->get_layer_res_size()*sizeof(float));
            oclw.add_variable(oclw_null,CL_READ_WRITE_CACHE,0);

            layers[i]->generate_kernels(loss);
            tmp=layers[i]->get_kernels_paths();
            kernels.insert(kernels.end(),tmp.begin(), tmp.end());

        }
    }




    if(device_index>-1)
    {
        std::sort(kernels.begin(), kernels.end());
        kernels.erase(std::unique(kernels.begin(), kernels.end()), kernels.end());

        int pos;
        for(std::string kernel : kernels)
        {
            pos=kernel.rfind("/");
            oclw.init_kernels({kernel.substr(pos+1)},kernel.substr(0,pos+1));
        }
    }

}

void NN::save(std::string file_name)
{
    size_t layers_data_size=0;
    auto ms=get_milliseconds_now();
    if(model_name!=default_model_name)
    {
        file_name=model_name;
    }

    if(logs_output)
        (*logs_output)<<"saving..."<<std::endl;

    for(int i=0; i<layers.size(); i++)
        layers_data_size+=layers[i]->get_layer_data_size();


    std::ofstream f(file_name, std::ios::out |std::ios::binary);//std::ios::binary
    f.write(reinterpret_cast<char*>(&layers_data_size), sizeof(size_t));

    for(int l=0; l<layers.size(); l++)
    {
        layers[l]->save(f);
        if(logs_output) (*logs_output)<<"saved layer: "<<l<<std::endl;
    }


    f<<std::endl<<"model data: "<<std::endl;
    for(int i=0; i<layers.size(); i++)
        f<<"layer index: "<<i
         <<"; neurons count: "<<layers[i]->get_neurons_count()<<std::endl;


    f.close();
    if(logs_output)
        (*logs_output)<<"SAVED"<<std::endl<<"ms spent: "<<get_milliseconds_now()-ms<<std::endl;
}

void NN::load(std::string file_name)
{
    size_t layers_data_size=0, file_layers_data_size;
    if(model_name!=default_model_name)
    {
        file_name=model_name;
    }
    std::ifstream f(file_name, std::ios::in | std::ios::binary);
    if(f.peek() == EOF)
    {
        if(logs_output)(*logs_output)<<"creating_new"<<std::endl;
        f.close();
        return;
    }

    for(int i=0; i<layers.size(); i++)
        layers_data_size+=layers[i]->get_layer_data_size();
    f.read(reinterpret_cast<char*>(&file_layers_data_size),sizeof(size_t));

    if(layers_data_size!=file_layers_data_size)
    {
        debug_utils::call_error(0,"load","the file size does not match the size of the model");
        f.close();
        return;
    }

    auto ms=get_milliseconds_now();
    if(logs_output)
        (*logs_output)<<"loading..."<<std::endl;

    for(int l=0; l<layers.size(); l++)
    {
        layers[l]->load(f);
        if(logs_output)
            (*logs_output)<<"loaded layer: "<<l<<std::endl;
    }
    f.close();

    if(logs_output)
        (*logs_output)<<"LOADED"<<std::endl<<"ms spent: "<<get_milliseconds_now()-ms<<std::endl;
}

std::string NN::predict_oclw(const std::string &input_key, int end_layer)
{
    predict_res_key=input_key;
    for(int i=layers.size()-1; i>=end_layer; i--)
    {
        predict_res_key=layers[i]->predict_oclw(predict_res_key);
    }
    return predict_res_key;
}


void NN::train_oclw(std::vector<size_t> &indexes, size_t &i)
{
    predict_oclw(nn_input_key+std::to_string(indexes[i]));


    layers[0]->calculate_ng_main_lay_oclw(
        (layers.size()==1)?nn_input_key+std::to_string(indexes[i]) : layers[1]->get_layer_res_key()
        ,nn_output_key+std::to_string(indexes[i]));



    for(int l_i=1; l_i<layers.size(); l_i++)
        layers[l_i]->calculate_ng_oclw((l_i==layers.size()-1? nn_input_key+std::to_string(indexes[i]):layers[l_i+1]->get_layer_res_key()));


    for(int l_i=0; l_i<layers.size(); l_i++)
        optimizer->update_params_oclw(layers[l_i],i);


}


void NN::correct_weights(const std::vector<float> &input, const std::vector<float> &output)
{
    if(layers.size()==0)
    {
        debug_utils::call_warning("NN::train","there are no layers in the neural network");
        return;
    }

    if(oclw.is_inited())
    {
        //oclw.write_variable(nn_input_key,input.size()*sizeof(float),const_cast<float*>(input.data()));
        //this line is called in predict func, and we use the data that is set there.

        oclw.write_variable(nn_output_key,output.size()*sizeof(float),const_cast<float*>(output.data()));


        layers[0]->calculate_ng_main_lay_oclw(
            (layers.size()==1)?nn_input_key : layers[1]->get_layer_res_key()
            ,nn_output_key);


        for(int l_i=1; l_i<layers.size(); l_i++)
            layers[l_i]->calculate_ng_oclw((l_i==layers.size()-1? nn_input_key:layers[l_i+1]->get_layer_res_key()));


        for(int l_i=0; l_i<layers.size(); l_i++)
            optimizer->update_params_oclw(layers[l_i],-1);

        return;
    }

    layers[0]->calculate_ng_main_lay(loss, layers.size()==1? input.data():layers[1]->get_layer_res(),  output.data());

    for(int l_i=1; l_i<layers.size(); l_i++)
        layers[l_i]->calculate_ng((l_i==layers.size()-1? input.data():layers[l_i+1]->get_layer_res()));

    for(int l_i=0; l_i<layers.size(); l_i++)
        optimizer->update_params(layers[l_i],-1);
}


std::vector<float> NN::predict(const std::vector<float> &input, int end_layer)
{
    if(layers.size()==0)
    {
        debug_utils::call_warning("NN::train","there are no layers in the neural network");
        return predict_res;
    }
    if(oclw.is_inited())
    {
        oclw.write_variable(nn_input_key,input.size()*sizeof(float),const_cast<float*>(input.data()));

        predict_res.resize(layers[end_layer]->get_layer_res_size());

        oclw.read_variable(
            predict_oclw(nn_input_key,end_layer),
            predict_res.size()*sizeof(float),predict_res.data());


        return predict_res;
    }


    predict_res=input;
    for(int i=layers.size()-1; i>=end_layer; i--)
    {
        predict_res=layers[i]->predict(predict_res);
    }

    return predict_res;
}



void NN::train(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output, int test_data_count, int epochs, int autosave, bool enable_mix_data)
{
    if(layers.size()==0)
    {
        debug_utils::call_warning("NN::train","there are no layers in the neural network");
        return;
    }


    std::vector<size_t> indexes;
    std::vector<std::vector<float>> test_input, test_output;
    float error, sum_error, max_error;



    if(logs_output)
    {
        if(test_data_count==-1)
        {
            test_input=input;
            test_output=output;
        }
        else if(test_data_count>0)
        {
            test_input.resize(test_data_count);
            test_output.resize(test_data_count);

            std::copy(input.end()-test_data_count, input.end(), test_input.begin());
            std::copy(output.end()-test_data_count,output.end(), test_output.begin());
            input.erase(input.end()-test_data_count, input.end());
            output.erase(output.end()-test_data_count, output.end());


        }
    }


    indexes.resize(input.size());
    for(size_t i=0; i<indexes.size(); ++i)indexes[i]=i;


    if(oclw.is_inited())
    {
        //copying train data to device

        for(size_t i=0; i<input.size(); i++)
        {
            oclw.add_and_write_variable(nn_input_key+std::to_string(i),CL_READ_ONLY_CACHE, input[i].size()*sizeof(float),input[i].data());
            oclw.add_and_write_variable(nn_output_key+std::to_string(i),CL_READ_ONLY_CACHE, output[i].size()*sizeof(float),output[i].data());
        }

        for(size_t i=0; i<test_input.size(); i++)
        {
            oclw.add_and_write_variable(nn_test_input_key+std::to_string(i),CL_READ_ONLY_CACHE, test_input[i].size()*sizeof(float),test_input[i].data());
        }


    }


    for(int epoch=0; epoch<epochs; epoch++)
    {
        auto ms=get_milliseconds_now();

        if(enable_mix_data) shuffle(indexes.begin(), indexes.end(), std::default_random_engine(rand()));


        for(size_t i=0; i<indexes.size(); i++)
        {
            if(oclw.is_inited())
            {
                train_oclw(indexes,i);
                continue;
            }


            predict(input[indexes[i]]);


            layers[0]->calculate_ng_main_lay(loss, layers.size()==1?input[indexes[i]].data():layers[1]->get_layer_res(), output[indexes[i]].data());

            for(int l_i=1; l_i<layers.size(); l_i++)
                layers[l_i]->calculate_ng((l_i==layers.size()-1? input[indexes[i]].data():layers[l_i+1]->get_layer_res()));


            for(int l_i=0; l_i<layers.size(); l_i++)
                optimizer->update_params(layers[l_i],i);
        }




        if(logs_output)
        {
            sum_error=0;
            max_error=0;
            for(int i=0; i<test_input.size(); i++)
            {
                if(oclw.is_inited())
                {
                    predict_res.resize(layers[0]->get_layer_res_size());
                    oclw.read_variable(predict_oclw(nn_test_input_key+std::to_string(i)),layers[0]->get_layer_res_size()*sizeof(float),predict_res.data());
                }
                else
                    predict_res=predict(test_input[i]);


                error=loss->calculate_error(test_output[i].data(),predict_res.data(),predict_res.size());
                max_error=std::max(error, max_error);
                sum_error+=error;
            }

            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);

            std::cout<<"epoch: "<<epoch<<"    ";
            if(test_data_count!=0)std::cout<<"error: "<<sum_error/test_input.size()<<"    "<<"max_error: "<<max_error<<"     ";
            std::cout<<"lr: "<<optimizer->get_learning_rate()<<"    ";
            std::cout<<"ms spent: "<<get_milliseconds_now()-ms
                     <<" ("<<std::put_time(&tm, "%d.%m.%Y; %H:%M:%S")<<")"<<std::endl;
        }

        optimizer->reduce_lr();

        if(autosave!=0 && epoch%autosave==0)save();
    }



    if(oclw.is_inited())
    {
        //deleting train data from device
        oclw.delete_variable(nn_test_input_key);
        for(size_t i=0; i<input.size(); i++)
        {
            oclw.delete_variable(nn_input_key+std::to_string(i));
            oclw.delete_variable(nn_output_key+std::to_string(i));
        }

        for(size_t i=0; i<test_input.size(); i++)
        {
            oclw.delete_variable(nn_test_input_key+std::to_string(i));
        }
    }
}









