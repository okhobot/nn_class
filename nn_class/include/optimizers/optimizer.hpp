#pragma once

#include <neuron.h>
#include <oclw.hpp>
#include <layers/Layer.hpp>
#include <Kernels_manager.hpp>

class Optimizer
{
protected:
    Kernels_manager km;

    float learning_rate, regularization_coef;
    float lr_reduction_coef, min_lr, max_lr;
    nn_type *weights_ptr,*gradients_ptr;

    OCLW *oclw=0;
    neuron *neurons_ptr;

public:
    Optimizer(float a_learning_rate=0.1, float a_lr_reduction_coef=1, float a_regularization_coef=0, float a_min_lr=1e-7,float a_max_lr=5e-1) ;
    // lr_reduction_coef - coefficient of change of learning_rate
    // a_min_lr - fuzzy minimal learning_rate
    // a_max_lr - fuzzy minimum learning_rate

    virtual void update_params(Layer *layer, const nn_size_type &data_index);

    virtual void update_params_oclw(Layer *layer, const nn_size_type &data_index);

    virtual void reduce_lr()
    {
        if(learning_rate>min_lr && learning_rate<max_lr)learning_rate/=lr_reduction_coef;
    }


    //setters
    virtual void set_oclw(OCLW *a_oclw)
    {
        oclw=a_oclw;
    }

    //getters

    std::vector<std::string> get_kernels_paths()
    {
        return km.get_kernels_paths();
    }

    float get_learning_rate()
    {
        return learning_rate;
    }
};
