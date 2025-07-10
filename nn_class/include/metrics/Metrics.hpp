#pragma once

#include<vector>
#include<string>

class Metrics
{
protected:
    std::string name="none";
public:
    inline std::string get_name()
    {
        return name;
    }

    virtual void reset() {}
    virtual void check(const std::vector<float> &predicted, const std::vector<float> &labels) {}
    virtual float get_result()
    {
        return 0;
    }
};
