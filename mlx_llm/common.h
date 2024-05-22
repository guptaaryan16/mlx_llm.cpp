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
    std::string name;
    std::unordered_map<std::string, array> parameters{};
    std::unordered_map<std::string, array> buffers{};
    std::unordered_map<std::string, Module &> submodules{};
    std::unordered_map<std::string, array &> named_parameters_dict{};

    StreamOrDevice device = metal::is_available() ? Device::gpu : Device::cpu;

    Module(){};
    Module(const Module &) = default;
    // Module& operator=(Module&&) noexcept = default;
    //  Module(Module&&) noexcept = default;

    virtual ~Module() = default;

    array &register_parameter(std::string name, array &wb);
    
    array &register_parameter(std::string name, array &&wb);
 
    array &register_buffer(std::string name, array &&wb);

    template <typename T>
    void register_module(std::string sub_name, T &m);

    template <typename T>
    void register_module(std::string sub_name, T &&m);

    template <typename T>
    void register_layer(std::string layers_name, std::vector<T> &layers);

    // Forward method for all submodules
    // TODO:: Make A general method for all forward implementations
    virtual array forward(const array &input);

    void named_parameters(std::string prelimiter = "");
    void update(std::unordered_map<std::string, array> trained_weights);

    void load_from_safetensors(const std::string &file, StreamOrDevice s);
    void load_from_gguf(const std::string &file, StreamOrDevice s);

    void load_weights(
        const std::string &file,
        StreamOrDevice s = metal::is_available() ? Device::gpu : Device::cpu);
    void print_parameters();
};

} // namespace mlx::core::nn
