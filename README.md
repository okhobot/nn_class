# nn_class
A modular neural network class with basic functionality
- It is recommended to use the release branch 

## Features
- Uses opencl_wrapper [https://github.com/okhobot/opencl_wrapper]

### contains
 - Sigmoid_activation
 - ReLU_activation
 - LeakyReLU_activation
 
 - MSE loss func
 - MAE loss func
 - LogLoss func

 - SGD_optimizer

 - Fully_connected_layer
 - Convolutional_layer
 - Image_resize_layer


## Usage
To create a neural network structure, you must first create instances of layers and hyperparameters, then create an instance of NN class and pass references to instances of the optimizer and loss functions to it. 
after initializing NN, add layers, starting with the output and ending with the input. After that, you need to call the init function to build the neural network.

- You can set oclw mode(parallel computing mode), if you set oclw device index in init func
- Devices and their indexes can be obtained using the get_available_devices_names() method
- The set_logs_output method is used to set the log stream




## Examples
- An example of using the library with MNIST dataset is described in the file src/example.cpp in the main branch
- MNIST dataset files are required [https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-images.idx3-ubyte] [https://www.kaggle.com/datasets/hojjatk/mnist-dataset?select=t10k-labels.idx1-ubyte]

## Сompilation 
- Make sure that the opencl_wrapper library is configured for the project
- For the compiled project to work using OpenCL, the kernel repository and the neuron file are required.h follow the path /include/neuron.h in the project repository
### An example of the structure of an assembled project
- nn_class.exe
- include/
    - neuron.h
- kernels/
    - ...
- t10k-labels.idx1-ubyte
- t10k-images.idx3-ubyte


## License
GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007