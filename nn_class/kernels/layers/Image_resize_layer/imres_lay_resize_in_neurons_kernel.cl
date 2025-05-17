#include <include/neuron.h>
__kernel void imres_lay_resize_in_neurons_kernel(
    __global const float *img_from,
    __global neuron *neurons_to,//img_to
    const float resize_coef_x,
    const float resize_coef_y,
    const float data_normalization_coef,

    const long new_x_size,
    const long new_y_size,
    const long frame_size,

    const long channels_count,
    const long old_y_size,
    const long old_x_size
)
{
    long channel = get_global_id(0); // getting the cycle index
    long i = get_global_id(1);
    long j = get_global_id(2);

    if(channel>=channels_count)return; // checking for out of bounds of array
    if(i>=old_y_size)return;
    if(j>=old_x_size)return;


    //printf("ok %f \n",img_from[channel*old_x_size*old_y_size+i*old_x_size+j]);


    if(resize_coef_y<1 && convert_int_rtn(i*resize_coef_y)==convert_int_rtn((i-1)*resize_coef_y))return;
    if(resize_coef_x<1 && convert_int_rtn(j*resize_coef_x)==convert_int_rtn((j-1)*resize_coef_x))return;

    long last_y,last_x;
    for(last_y=i+1;convert_int_rtn(last_y*resize_coef_y)==convert_int_rtn((last_y-1)*resize_coef_y);++last_y);
    for(last_x=j+1;convert_int_rtn(last_x*resize_coef_x)==convert_int_rtn((last_x-1)*resize_coef_x);++last_x);
    last_x=min(last_x, old_x_size);
    last_y=min(last_y, old_y_size);

    float res_x=frame_size+j*resize_coef_x,
    res_y=frame_size+i*resize_coef_y;

    //printf("ok %d\n",(int)(channel*new_x_size*new_y_size+((int)i*resize_coef_y)*new_x_size+ ((int)j*resize_coef_x)));

    for(int y=i;y<last_y ;y++)
    {
        for(int x=j;x<last_x;x++)
            {
                res_x=frame_size+x*resize_coef_x;

                neurons_to[channel*new_x_size*new_y_size+((int)res_y)*new_x_size+ ((int)res_x)].gradient+=img_from[channel*old_x_size*old_y_size+y*old_x_size+x]*data_normalization_coef;

                //printf("ok %lld %lld %d \n",i,j,(int)channel*new_x_size*new_y_size+((int)res_y)*new_x_size+ ((int)res_x));

                if(resize_coef_x>1||resize_coef_y>1)
                {
                    float fill_e=neurons_to[channel*new_x_size*new_y_size+((int)res_y)*new_x_size+ ((int)res_x)].gradient;

                    for(int dy=(int)res_y;dy<(int)(res_y+resize_coef_y);++dy)
                        for(int dx=(int)res_x;dx<(int)(res_x+resize_coef_x);++dx)
                            neurons_to[channel*new_x_size*new_y_size+dy*new_x_size+ dx].gradient=fill_e;
                }
            }
        res_y+=resize_coef_y;
    }


    //printf("ok \n");





}
