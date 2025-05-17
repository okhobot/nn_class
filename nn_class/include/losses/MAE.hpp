#pragma once

#include <losses/Loss.hpp>

class MAE : public Loss
{
protected:



public:

    MAE()
    {
        inline_add_loss_gradient="(output[i]-layer_res[i]>0?1:(output[i]-layer_res[i]<0?-1:0))";
    }

    float calculate_error(float *output, float *layer_res, const nn_size_type &output_size) override
    {
        float res=0;
        for(nn_size_type i=0; i<output_size; i++)
            res+=abs(output[i]-layer_res[i]);
        return res;
    }

    void add_loss_gradient(const float &output, const float &layer_res, neuron *neuron)override
    {
        a_neuron->gradient+=(output-layer_res>0?1:(output-layer_res<0?-1:0));
    }
};
