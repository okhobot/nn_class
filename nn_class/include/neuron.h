#ifndef NEURON_H_
#define NEURON_H_

#ifdef __OPENCL_VERSION__
typedef long nn_size_type; // 8 bytes for cl
#else
typedef long long nn_size_type; //8 bytes for cpp
inline float max(float a, float b)
{
    return a>=b?a:b;
}
inline float min(float a, float b)
{
    return a<=b?a:b;
}
#endif

typedef float nn_type;
#define nn_t_bounds 700 //700 for float; 1e+9 for int, 126 for char;
#define nn_type_translate_coef 1.0 //1 for float, 1e+7 for int, 100 for char

#define oclw_null "-"

inline float tof(const nn_type n)//from nn_type to float
{
    return n/nn_type_translate_coef;
}
inline float fromf(const float n)//from float to nn_type
{
    return n*nn_type_translate_coef;
}
inline void add(nn_type *val, float n)//add n to val, taking into account the boundaries
{
    *val+= (nn_type)max((float)(-nn_t_bounds-*val),min((float)(nn_t_bounds-*val), n));
}


typedef struct neuron
{
    nn_size_type params_start_index;
    nn_size_type params_count;

    nn_type b;

    float b_grad;
    float gradient;
} neuron;

#endif
