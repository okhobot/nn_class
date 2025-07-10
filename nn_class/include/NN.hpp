#pragma once

#include <neuron.h>
#include <oclw.hpp>
#include <optimizers/Optimizer.hpp>
#include <losses/Loss.hpp>
#include <metrics/Metrics.hpp>

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>

#define default_model_name "model.dat"

class NN
{

protected:



    const std::string nn_input_key="nn_input",
                      nn_test_input_key="nn_test_input",
                      nn_output_key="nn_output";

    OCLW oclw;
    Optimizer *optimizer_ptr;
    Loss *loss_ptr;
    Metrics *metric_ptr;

    std::ostream *logs_output=0;

    std::vector<Layer*> layers;
    std::vector<float> predict_res;
    std::string predict_res_key;
    std::string model_name=default_model_name;


    long long int get_milliseconds_now()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }


    std::string predict_oclw(const std::string &input_key, int end_layer=0);

    void train_oclw(std::vector<size_t> &indexes,size_t &i);

public:
    NN(Optimizer *a_optimizer_ptr, Loss *a_loss_ptr, Metrics *a_metric_ptr=0);

    virtual void show();// print model data to logs_out

    // setters
    void set_model_name(std::string name)
    {
        model_name=name;
    }

    void set_logs_output(std::ostream *out)
    {
        logs_output=out;
        oclw.set_logs_output(logs_output);
    }

    // add layer
    // this should be called from the output layer to the input layer.
    virtual void add_layer(Layer *layer)
    {
        layers.push_back(layer);
    }
    // getters
    std::vector<std::string> get_available_devices_names()
    {
        return oclw.get_available_devices_names();
    }

    virtual void init(int device_index=-1, unsigned seed=9);// init nn
    // device_index - OpenCl device index
    // seed - seed for weights generation

    virtual void correct_weights(const std::vector<float> &input, const std::vector<float> &output);// for reinforcement learning
    // first, call predict
    // then call it with same input

    virtual std::vector<float> predict(const std::vector<float> &input, int end_layer=0);// end layer - index of output layer

    virtual void train(std::vector<std::vector<float>> input, std::vector<std::vector<float>> output, int test_data_count=0, int epochs=50, int autosave=0,bool enable_mix_data=true);
    // test_data_count - count of examples in train data for test(it will not be used in training)
    // autosave - the number of epochs to save the model. if 0, no autosave


    virtual void load(std::string file_name=default_model_name);

    virtual void save(std::string file_name=default_model_name);

};
