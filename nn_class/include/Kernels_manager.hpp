#pragma once

#include <debug_utils.hpp>

#include <map>
#include <vector>
#include <iostream>

class Kernels_manager
{
    std::map<std::string, std::string> kernels;
    std::string default_path;

public:

    inline void set_default_path(std::string a_default_path)// set path to kernels
    {
        default_path=a_default_path;
    }

    inline void add_kernel(std::string name,std::string value, std::string path="none")// add kernel to dict
    {
        if(path=="none")path=default_path;
        kernels[name]=value;
        kernels["path_"+name]=path+value;
    }

    inline void delete_kernel(std::string name)// delete kernel from dict
    {
        kernels.erase(name);
        kernels.erase("path_"+name);
    }

    inline std::vector<std::string> get_kernels_paths()
    {
        std::vector<std::string> res;
        for(std::map<std::string, std::string>::iterator it = kernels.begin(); it != kernels.end(); ++it)
        {
            if(it->first.substr(0, 4)=="path")res.push_back(it->second);
        }
        return res;
    }

    inline std::string get(const std::string &name)// get kernel name by variable name
    {
        if(kernels[name]=="")debug_utils::call_error(1,"Kernel_manager::get","no_kernel_with_such_key",name);
        return kernels[name];
    }

    inline bool have(const std::string &name)// get kernel name by variable name
    {
        return kernels.find(name) != kernels.end();
    }

    inline std::string get_path(const std::string &name)// get kernel path by variable name
    {
        if(kernels[name]=="")debug_utils::call_error(1,"Kernel_manager::get","no_kernel_with_such_key",name);
        return kernels["path_"+name];
    }
};
