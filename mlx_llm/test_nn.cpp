#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "utils.cpp"

using namespace mlx::core;

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

    array &register_buffer(std::string name, array &&wb)
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
            submodules.insert({sub_name, m});
            submodules.at(sub_name).name = sub_name;
            submodules.at(sub_name).named_parameters(sub_name);
        }
        else
        {
            // Handle the case where the submodule doesn't exist
        }
    }

    template <typename T>
    void register_module(std::string sub_name, T &&m)
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
            submodules.insert({sub_name, m});
            submodules.at(sub_name).name = sub_name;
            submodules.at(sub_name).named_parameters(sub_name);
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
        for (auto &[k, v] : this->parameters)
        {
            const std::string sub_name = get_name(prelimiter, k);
            std::cout << this->name << ":" << sub_name << std::endl;
            this->named_parameters_dict.insert({sub_name, v});
        }
        for (auto &[k, v] : this->buffers)
        {
            const std::string sub_name = get_name(prelimiter, k);
            this->named_parameters_dict.insert({sub_name, v});
        }
        if (!this->submodules.empty())
        {
            for (auto &[k, v] : this->submodules)
            {
                for (auto &[l, m] : v.named_parameters_dict)
                {
                    const std::string sub_name = get_name(prelimiter, l);
                    std::cout << this->name << ":" << sub_name << std::endl;
                    // std::cout << m << std::endl;
                    if (!(endsWith(l, ".")))
                    {
                        this->named_parameters_dict.insert({sub_name, m});
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

class LinearLayer : public Module
{
public:
    int input_dim, output_dim;
    bool with_bias = true;

    LinearLayer() = default;
    LinearLayer(const LinearLayer &) = default;
    LinearLayer(int in_features, int out_features, bool _with_bias = true)
    {
        input_dim = in_features;
        output_dim = out_features;
        array weight = random::normal({in_features, out_features}, float32);
        array bias = random::normal({out_features}, float32);
        with_bias = _with_bias;

        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    ~LinearLayer() = default;

    array forward(const array &input) override
    {
        // Check if input size matches number of weights in first layer
        if (input.shape(-1) != parameters.at("weight").shape(0))
        {
            throw std::invalid_argument(
                "Input size doesn't match weight vector size");
        }
        // Allocate space for the outputs
        array outputs = matmul(input, parameters.at("weight"));

        return with_bias ? (outputs + parameters.at("bias")) : outputs;
    }
};

class LinearOnesLayer : public Module
{
public:
    int input_dim, output_dim;
    bool with_bias = true;

    LinearOnesLayer() = default;
    LinearOnesLayer(const LinearOnesLayer &) = default;
    LinearOnesLayer(int in_features, int out_features, bool _with_bias = true)
    {
        input_dim = in_features;
        output_dim = out_features;
        array weight = ones({in_features, out_features}, float32);
        array bias = ones({out_features}, float32);
        with_bias = _with_bias;

        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    ~LinearOnesLayer() = default;

    array forward(const array &input) override
    {
        // Check if input size matches number of weights in first layer
        if (input.shape(-1) != parameters.at("weight").shape(0))
        {
            throw std::invalid_argument(
                "Input size doesn't match weight vector size");
        }
        // Allocate space for the outputs
        array outputs = matmul(input, parameters.at("weight"));

        return with_bias ? (outputs + parameters.at("bias")) : outputs;
    }
};

class CustomLayer : public Module
{
public:
    int input_dim, output_dim;
    LinearOnesLayer l2;
    bool with_bias = true;

    CustomLayer() = default;
    CustomLayer(const CustomLayer &) = default;
    CustomLayer(int in_features, int out_features, bool _with_bias = true)
    {
        input_dim = in_features;
        output_dim = out_features;
        array weight = random::normal({in_features, out_features}, float32);
        array bias = random::normal({out_features}, float32);
        l2 = LinearOnesLayer(out_features, out_features);
        register_parameter("weight", weight);
        register_parameter("bias", bias);
        register_module("l1", l2);
        with_bias = _with_bias;
    }

    ~CustomLayer() = default;

    array forward(const array &input) override
    {
        // Check if input size matches number of weights in first layer
        if (input.shape(-1) != parameters.at("weight").shape(0))
        {
            throw std::invalid_argument(
                "Input size doesn't match weight vector size");
        }
        // Allocate space for the outputs
        array outputs = matmul(input, parameters.at("weight"));

        auto y = with_bias ? (outputs + parameters.at("bias")) : outputs;
        return l2.forward(y);
    }
};

class MyModel : public Module
{
public:
    LinearLayer fc1 = LinearLayer(784, 100, false);
    CustomLayer fc2 = CustomLayer(100, 10, false);
    std::vector<CustomLayer> layers{};

    MyModel()
    {
        for (int i = 0; i < 3; i++)
        {
            layers.push_back(CustomLayer(10, 10));
        }
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_layer("layers", layers);
    }
    array forward(const array &x) override
    {
        auto y = fc2.forward(fc1.forward(x));
        for (auto &l : layers)
        {
            y = l.forward(y);
        }
        return y;
    }
};

int main()
{
    array input = random::uniform({1, 784});
    MyModel model = MyModel();
    auto res = model.forward(input);
    std::cout << res << "\n"
              << res.shape();
    model.print_parameters();
    model.load_weights(
        "/Users/guptaaryan16/Desktop/OSS/mlx/examples/cpp/mymodel.safetensors");
    auto res_ = model.forward(input);
    model.print_parameters();
    std::cout << res_ << "\n"
              << res_.shape();
    return 0;
}