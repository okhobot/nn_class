#pragma once

#include <losses/Loss.hpp>

#include<cmath>

class LogLoss : public Loss
{
protected:



public:

    LogLoss()
    {
        inline_add_loss_gradient="((output[i]-layer_res[i])/(layer_res[i]*(1-layer_res[i])))";
    }

    virtual float calculate_error(float *output, float *layer_res, const nn_size_type &output_size)override
    {
        float res=0;
        for(nn_size_type i=0; i<output_size; i++)
            res-=output[i]*log(layer_res[i])+(1-output[i])*log(1-layer_res[i]);
        return res;
    }

    virtual void add_loss_gradient(const float &output, const float &layer_res, neuron *neuron)override
    {
        neuron->gradient+=(output-layer_res)/(layer_res*(1-layer_res));
    }
};
