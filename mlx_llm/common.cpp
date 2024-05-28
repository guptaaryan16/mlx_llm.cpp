#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "utils.cpp"

namespace mlx::core::nn{

    class Module
    {
    public:
        std::unordered_map<std::string, array> parameters{};
        std::unordered_map<std::string, array> buffers{};
        std::unordered_map<std::string, std::shared_ptr<Module>> submodules{};
        std::unordered_map<std::string, array &> named_parameters_dict{};

        std::string name;
        StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;

        Module(){};
        Module(const Module &) = default;
        // Module& operator=(Module&&) noexcept = default;
        //  Module(Module&&) noexcept = default;

        virtual ~Module() = default;

        array &register_parameter(std::string name, array &wb)
        {
            // `register_parameter` allows you to register the Weights & Biases
            // used by the NN
            parameters.insert({name, wb});
            return parameters.at(name);
        }

        array &register_parameter(std::string name, array &&wb)
        {
            // `register_parameter` allows you to register the Weights & Biases
            // used by the NN
            parameters.insert({name, wb});
            return parameters.at(name);
        }

        array &register_buffer(std::string name, array &wb)
        {
            // `register_parameter` allows you to register the Weights & Biases
            // used by the NN
            buffers.insert({name, wb});
            return buffers.at(name);
        }

        template <typename T>
        void register_module(std::string sub_name, T &m)
        {
            // `register_component` allows you to register the component(in order) as
            // used by the NN
            if (!std::is_base_of<T, Module>::value)
            {
                // Error the code is not correct
            }

            // Check if the submodule exists before trying to access it
            if (!(submodules.find(sub_name) != submodules.end()))
            {
                // Add the parameters of the submodules to the named_parameters_dict
                //   m.name = sub_name;
                //   m.named_parameters();
                submodules.insert({sub_name, m});
                submodules.at(sub_name)->name = sub_name;
                submodules.at(sub_name)->named_parameters();
            }
            else
            {
                // Handle the case where the submodule doesn't exist
            }
        }

        template <typename T>
        void register_module(std::string sub_name, std::shared_ptr<T> &&m)
        {
            // `register_component` allows you to register the component(in order) as
            // used by the NN
            if (!std::is_base_of<T, Module>::value)
            {
                // Error the code is not correct
            }

            // Check if the submodule exists before trying to access it
            if (!(submodules.find(sub_name) != submodules.end()))
            {
                // Add the parameters of the submodules to the named_parameters_dict
                // m.name = sub_name;
                // m.named_parameters();
                submodules.insert({sub_name, m});
                submodules.at(sub_name)->name = sub_name;
                submodules.at(sub_name)->named_parameters();
            }
            else
            {
                // Handle the case where the submodule doesn't exist
            }
        }

        template <typename T>
        void register_layer(std::string layers_name, std::vector<T> &layers)
        {
            // `register_component` allows you to register the layers(in order) as
            // used by the NN
            if (!std::is_base_of<T, Module>::value)
            {
                // Error the code is not correct
            }
            for (size_t i = 0; i < layers.size(); i++)
            {
                register_module(get_name(layers_name, i), layers[i]);
            }
        }

        // Forward method for all submodules
        // TODO:: Make A general method for all forward implementations
        virtual array forward(const array &input)
        {
            return input;
        };

        void named_parameters(std::string prelimiter = "")
        {
            for (auto &[k, v] : parameters)
            {
                std::string sub_name = get_name(prelimiter, name, k);
                named_parameters_dict.insert({sub_name, v});
            }
            for (auto &[k, v] : buffers)
            {
                std::string sub_name = get_name(prelimiter, name, k);
                named_parameters_dict.insert({sub_name, v});
            }
            if (!submodules.empty())
            {
                for (auto &[k, v] : submodules)
                {
                    for (auto &[l, m] : v->named_parameters_dict)
                    {
                        std::string sub_name = get_name(name, l);
                        if (!(endsWith(sub_name, ".")))
                        {
                            named_parameters_dict.insert({sub_name, m});
                        }
                    }
                }
            }
        }

        void update(std::unordered_map<std::string, array> trained_weights)
        {
            // Create references for all the known parameters
            this->named_parameters();

            for (auto &[k, v] : trained_weights)
            {
                if (!(named_parameters_dict.find(k) != named_parameters_dict.end()))
                {
                    std::cout << "Named parameter does not contain the key: " << k << "\n";
                }
                else if (named_parameters_dict.at(k).shape() != v.shape())
                {
                    std::cout << "There is a shape difference for : " << k << "->"
                              << named_parameters_dict.at(k).shape() << " and " << v.shape()
                              << std::endl;
                }
                else
                {
                    named_parameters_dict.at(k) = v;
                }
            }
        }

        void load_from_safetensors(const std::string &file, StreamOrDevice s)
        {
            SafetensorsLoad loaded_weights = load_safetensors(file, s);
            update(loaded_weights.first);
        }

        void load_from_gguf(const std::string &file, StreamOrDevice s)
        {
            GGUFLoad loaded_weights = load_gguf(file, s);
            update(loaded_weights.first);
        }

        void load_weights(
            const std::string &file,
            StreamOrDevice s = metal::is_available() ? Device::gpu : Device::cpu)
        {
            if (endsWith(file, ".safetensors"))
            {
                std::cout << "Loading model from .safetensors file...\n";
                load_from_safetensors(file, s);
            }
            else if (endsWith(file, ".gguf"))
            {
                load_from_gguf(file, s);
                std::cout << "Loading model from .gguf file...\n";
            }
            else
            {
                std::cout << "Model file format is not supported...\n";
            }
        }

        void print_parameters()
        {
            this->named_parameters();
            std::cout << "\n[\nparameters:\n";
            for (auto &[k, v] : named_parameters_dict)
            {
                std::cout << k << ":\n"
                          << v << "\n";
            }
            std::cout << "]\n";
        }
    };

} //namespace mlx::core::nn