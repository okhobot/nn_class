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
To create a neural network structure, you must first create instances of layers and hyperparameters, then create an instance of class T and pass references to instances of the optimizer and loss functions to it. 
after initializing NN, add layers, starting with the output and ending with the input. After that, you need to call the init function to build the neural network.

- You can set oclw mode(parallel computing mode), if you set oclw device index in init func
- Devices and their indexes can be obtained using the get_available_devices_names() method
- The set_logs_output method is used to set the log stream


## Examples
- An example of using the library with MNIST dataset is described in the file example.cpp in the main branch


## License
GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007