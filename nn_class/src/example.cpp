///EXAMPLE


#include <iostream>
#include <vector>

#include <NN.hpp>
#include <layers/Image_resize_layer.hpp>
#include <layers/Convolutional_layer.hpp>
#include <layers/Dense_layer.hpp>
#include <optimizers/SGD_optimizer.hpp>
#include <activations/LeakyReLU_activation.hpp>
#include <activations/Sigmoid_activation.hpp>
#include <activations/Softmax_activation.hpp>
#include <losses/LogLoss.hpp>
#include<metrics/Multiclass_accuracy.hpp>

int convert_int (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_MNIST_images(std::vector<std::vector<float>> &arr, std::string path)
{
    std::ifstream file (path,std::ios::binary);
    if (file.is_open())
    {
        int magic_number;
        int images_count;
        int n_rows;
        int n_cols;

        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= convert_int(magic_number);

        file.read((char*)&images_count,sizeof(images_count));
        images_count= convert_int(images_count);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= convert_int(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= convert_int(n_cols);

        arr.resize(images_count,std::vector<float>(n_rows*n_cols));

        for(int i=0; i<images_count; ++i)
        {
            for(int r=0; r<n_rows; ++r)
            {
                for(int c=0; c<n_cols; ++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= temp/255.0;
                }
            }
        }
    }
    else
        debug_utils::call_error(1,"read_MNIST_images","file not found", "file name: "+path);
}

void read_MNIST_labels(std::vector<std::vector<float>> &arr, std::string path)
{

    std::ifstream file (path,std::ios::binary);
    if (file.is_open())
    {
        int magic_number;
        int labels_count;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
        magic_number = convert_int(magic_number);


        file.read(reinterpret_cast<char*>(&labels_count), sizeof(int));
        labels_count = convert_int(labels_count);

        arr.resize(labels_count,std::vector<float>(1));

        for(int i=0; i<labels_count; ++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            arr[i][0]= temp;
        }
    }
    else
        debug_utils::call_error(1,"read_MNIST_labels","file not found", "file name: "+path);
}


std::vector<std::vector<float>> convert_output(std::vector<std::vector<float>> old_output)
{
    std::vector<std::vector<float>> res;
    res.resize(old_output.size(),std::vector<float>(10));
    for(int i=0; i<old_output.size(); i++)
        res[i][old_output[i][0]]=1;
    return res;
}



int main()
{
    //reading dataset
    std::vector<std::vector<float>> input, output;
    read_MNIST_images(input,"train-images.idx3-ubyte");
    read_MNIST_labels(output,"train-labels.idx1-ubyte");

    output=convert_output(output);

    //creating nn params
    SGD_optimizer opt(16,0.0001,1.05,0,1e-7);
    LogLoss loss;
    Multiclass_accuracy accuracy;

    LeakyReLU_activation l_relu;
    Softmax_activation softmax;
    Sigmoid_activation sigmoid;

    //creating nn layers
    Convolutional_layer input_lay(&l_relu,28,28,1,32);//32 conv. kernels
    Image_resize_layer irl(26,26,26,26,32,1);//add frame with size 1 pix to output image
    Dense_layer hide_lay(&l_relu,irl.get_layer_res_size(),128,-2);// input size - from irl output size, 128 neurons
    Dense_layer out_lay(&softmax,128,10,-2);// input size - 128, output - 10


    NN nn(&opt,&loss, &accuracy);//create nn with nn params
    nn.add_layer(&out_lay);// adding layers
    nn.add_layer(&hide_lay);// from output to input
    nn.add_layer(&irl);
    nn.add_layer(&input_lay);

    std::vector<std::string> device_names =nn.get_available_devices_names();//get available for openCl devices(with indexes)
    for(int i=0; i<device_names.size(); i++)//print it
        std::cout<<device_names[i]<<" ";
    std::cout<<std::endl;

    nn.set_logs_output(&std::cout);//set logs output to console

    nn.init();//init nn
    //you can set oclw mode(parallel computing mode), if you set oclw device index


    std::cout << std::endl;
    nn.show();//show nn data
    std::cout << std::endl;

    nn.train(input,output,1000,2,0,false);//train nn
    //1000 dataset examples for test

    //print examples with nn predicts
    for(int j=input.size()-1; j>=input.size()-6; j--)
    {
        std::cout<<"expected: "<<std::distance(output[j].begin(),std::find(output[j].begin(),output[j].end(),1))<<std::endl;
        std::cout<<"result: ";
        std::vector<float> res=nn.predict(input[j]);//predicting
        for(int i=0; i<res.size(); i++)std::cout << res[i] << " ";
        std::cout<<std::endl;
    }


    return 0;
}
