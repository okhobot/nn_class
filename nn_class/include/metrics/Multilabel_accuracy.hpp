#pragma once

#include<metrics/Metrics.hpp>
#include<neuron.h>
#include<algorithm>

class Multilabel_accuracy :public Metrics
{
    nn_size_type tests_count, passed_tests_count;
    float threshold;
public:
    Multilabel_accuracy(float a_threshold=0.5)
    {
        name="ml_accuracy";

        tests_count=0;
        passed_tests_count=0;

        threshold=a_threshold;
    }
    inline void reset() override
    {
        tests_count=0;
        passed_tests_count=0;
    }
    inline void check(const std::vector<float> &predicted, const std::vector<float> &labels)override
    {
        ++tests_count;
        for(int i=0; i<predicted.size(); i++)
            if((predicted[i]>=threshold)!=(bool)labels[i])
                return;
        ++passed_tests_count;
    }
    inline float get_result()override
    {
        return (float)passed_tests_count/tests_count;
    }
};
