#pragma once

#include<metrics/Metrics.hpp>
#include<neuron.h>
#include<algorithm>

class Multiclass_accuracy :public Metrics
{
    nn_size_type tests_count, passed_tests_count;
public:
    Multiclass_accuracy()
    {
        name="mc_accuracy";
        tests_count=0;
        passed_tests_count=0;
    }
    inline void reset() override
    {
        tests_count=0;
        passed_tests_count=0;
    }
    inline void check(const std::vector<float> &predicted, const std::vector<float> &labels)override
    {
        ++tests_count;
        if(std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end())) ==
                std::distance(labels.begin(), std::max_element(labels.begin(), labels.end())) )
            ++passed_tests_count;
    }
    inline float get_result()override
    {
        return (float)passed_tests_count/tests_count;
    }
};
