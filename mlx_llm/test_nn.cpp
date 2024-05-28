#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "utils.cpp"
#include "common.cpp"

using namespace mlx::core;
class LinearLayer : public nn::Module
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

class LinearOnesLayer : public nn::Module
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

class CustomLayer : public nn::Module
{
public:
    int input_dim, output_dim;
    std::shared_ptr<LinearLayer> l1;
    bool with_bias = true;

    CustomLayer() = default;
    CustomLayer(const CustomLayer &) = default;
    CustomLayer(int in_features, int out_features, bool _with_bias = true)
    {
        input_dim = in_features;
        output_dim = out_features;
        array weight = random::normal({in_features, out_features}, float32);
        array bias = random::normal({out_features}, float32);
        l1 = std::make_shared<LinearLayer>(out_features, out_features);

        register_parameter("weight", weight);
        register_parameter("bias", bias);
        register_module("l1", l1);
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
        return l1->forward(y);
    }
};
class TestModel : public nn::Module
{
public:
    std::shared_ptr<LinearLayer> fc1;
    std::shared_ptr<CustomLayer> fc2;
    std::vector<std::shared_ptr<CustomLayer>> layers{};

    TestModel()
    {
        // Declare your layers only inside the constuctor to avoid "undefined"
        // behaviour
        fc1 = std::make_shared<LinearLayer>(784, 100, false);
        fc2 = std::make_shared<CustomLayer>(100, 10, false);

        for (int i = 0; i < 3; i++)
        {
            layers.push_back(std::make_shared<CustomLayer>(10, 10));
        }

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_layer("layers", layers);
    }
    array forward(const array &x) override
    {
        auto y = fc2->forward(fc1->forward(x));
        for (auto &l : layers)
        {
            y = l->forward(y);
        }
        return y;
    }
};
