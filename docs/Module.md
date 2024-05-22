## Documentation 

#### Module API (defined as mlx::core::nn::Module)

To use the Module API, you can refer to the file `mlx_llm/common.cpp` where we have defined the API for a Module object.

Let us take an example of a simple Layer Definition i.e. LinearLayer.

```
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
```

Here you can see, that the simple class `LinearLayer` imports from the module nn::Module and then registers the `weight` and `bias` using `Module::register_parameter` method.
Similarly we can register non-trainiable methods using  `Module::register_buffer` method.

Further any new submodule will be registered as `Module::register_submodule` method.

Now you can create a `Module::forward` method that can take in arguments like an `mlx::core::array` and give an output.
