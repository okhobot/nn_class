#pragma once

#include <optimizers/Optimizer.hpp>

#include <vector>

class SGD_optimizer : public Optimizer
{
    int batch_size;
    bool update_weights;
public:
    SGD_optimizer(int a_batch_size=1, float a_learning_rate=0.1, float a_lr_reduction_coef=1, float a_min_lr=1e-7, float a_regularization_coef=0,OCLW *a_oclw=0);

    void update_params(Layer *layer, const nn_size_type &input_index)override;

    void update_params_oclw (Layer *layer, const nn_size_type &input_index)override;
};

