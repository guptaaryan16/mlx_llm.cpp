/*
Author: Aryan Gupta (guptaaryan16)
Based on https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/phi3.py && HF-MLX community Model
*/
#pragma once

#include <any>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include "mlx/mlx.h"
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

class Dropout : public nn::Module
{
public:
    float p = 0.5;
    float _p_1 = 1 - p;

    Dropout(const Dropout &) = default;
    Dropout(float _p = 0.5)
    {
        p = _p;
        _p_1 = 1 - _p_1;
    };

    array forward(array &x)
    {
        if (_p_1 == 1.0)
            return x;
        array mask = random::bernoulli(_p_1, x.shape());
        return (1 / _p_1) * mask * x;
    }
};
class RoPE : public nn::Module
{
public:
    int dims;
    bool traditional = false;
    float base = 10000;
    float scale = 1.0;

    RoPE() = default;
    RoPE(
        int dims,
        bool traditional = false,
        float base = 10000,
        float scale = 1.0)
    {
        this->dims = dims;
        this->traditional = traditional;
        this->base = base;
        this->scale = scale;
    }
    array forward(array x, int offset = 0)
    {
        return mlx::core::fast::rope(x, dims, traditional, base, scale, offset, device);
    }
};

class RMSNorm : public nn::Module
{
public:
    float eps;
    RMSNorm() = default;
    RMSNorm(int dims, float _eps = 1e-5)
    {
        eps = _eps;
        array weight = ones({dims}, float32);
        register_parameter("weight", weight);
    }

    array forward(array x)
    {
        return mlx::core::fast::rms_norm(
            x, parameters.at("weight"), eps);
    }
};

class Embedding : public nn::Module
{
public:
    Embedding() = default;
    Embedding(int dims, int num_embeddings)
    {
        array scale = array(sqrt(1 / dims));
        array weight = random::normal({num_embeddings, dims}, float32) * scale;

        register_parameter("weight", weight);
    }
    array forward(array x)
    {
        return matmul(parameters.at("weight"), x);
    }
};

array scaled_dot_product_attention(
    array queries,
    array keys,
    array values,
    float scale,
    const std::__1::optional<mlx::core::array> &mask = std::nullopt,
    StreamOrDevice s = metal::is_available() ? Device::gpu : Device::cpu)
{
    return mlx::core::fast::scaled_dot_product_attention(queries, keys, values, scale, mask, s);
}

array silu(array x)
{
    return x * sigmoid(x);
}

struct PhiModelConfig
{
    int num_hidden_layers;
    int vocab_size;
    int hidden_size;
    int rms_norm_eps;
    std::string model_type;
    int num_hidden_layer;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads = 0;
    float rope_theta = 10000;
    float rope_scale;
    bool rope_traditional = false;
    // <std::map<std::variant<float, std::string>> rope_scaling = nullptr;

    PhiModelConfig()
    {
        if (!num_key_value_heads)
            num_key_value_heads = num_attention_heads;
    }
};
class PhiAttention : public nn::Module
{
public:
    LinearLayer query_proj, key_proj, value_proj, out_project;
    int dim, n_heads, n_kv_head, head_dim, op_size;
    float scale, rope_scale;
    LinearLayer qkv_proj, o_proj;
    RoPE rope;

public:
    PhiAttention() = default;
    PhiAttention(struct PhiModelConfig args)
    {
        dim = args.hidden_size;
        n_heads = args.num_attention_heads;
        n_kv_head = args.num_key_value_heads;
        head_dim = int(args.hidden_size / n_heads);
        scale = pow(head_dim, -0.5);
        op_size = n_heads * head_dim + 2 * (n_kv_head * head_dim);
        qkv_proj = LinearLayer(dim, op_size, false);
        o_proj = LinearLayer(n_heads * head_dim, dim, false);

        rope_scale = 1;
        rope = RoPE(head_dim, args.rope_traditional, args.rope_theta, args.rope_scale);
        register_module("qkv_proj", qkv_proj);
        register_module("o_proj", o_proj);
    }
    array forward(array x, array &queries, array &keys, array &values, array &mask)
    {
        int B = queries.shape(0), L = queries.shape(1), D = queries.shape(2);
        array qkv = qkv_proj.forward(x);
        auto res = split(qkv, 3, -1);
        array _queries = res[0], _keys = res[1], _values = res[2];
        _queries = transpose(reshape(_queries, {B, L, n_kv_head, -1}), {0, 2, 1, 3});
        _keys = transpose(reshape(_keys, {B, L, n_kv_head}), {2, 3, 1});
        _values = transpose(reshape(_values, {B, L, n_heads, -1}), {0, 2, 1, 3});
        eval(_queries);
        eval(_keys);
        eval(_values);

        _queries = rope.forward(_queries);
        _keys = rope.forward(_keys, dim);

        array output = scaled_dot_product_attention(
            queries, keys, values, scale, mask);
        output = reshape(transpose(output, {0, 2, 1, 3}), {B, L, -1});
        return o_proj.forward(output);
    }
};

class MLP : public nn::Module
{
public:
    LinearLayer gate_up_proj, down_proj;

    MLP() = default;
    MLP(int dim, int hidden_dim)
    {
        gate_up_proj = LinearLayer(dim, 2 * hidden_dim, false);
        down_proj = LinearLayer(hidden_dim, dim, false);
        register_module("gate_up_proj", gate_up_proj);
        register_module("down_proj", down_proj);
    }
    array forward(array x)
    {
        x = gate_up_proj.forward(x);
        auto res = split(x, 2, -1);
        array gate = res[0], _x = res[1];
        return down_proj.forward(silu(gate) * _x);
    }
};

class TransformerBlock : public nn::Module
{
public:
    int num_attention_heads, hidden_size;
    PhiAttention self_attn;
    MLP mlp;
    RMSNorm input_layernorm, post_attention_layernorm;
    struct PhiModelConfig args;

    TransformerBlock() = default;
    TransformerBlock(struct PhiModelConfig _args)
    {
        args = _args;
        num_attention_heads = args.num_attention_heads;
        hidden_size = args.hidden_size;
        self_attn = PhiAttention(args);
        mlp = MLP(args.hidden_size, args.intermediate_size);
        input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps);
        post_attention_layernorm =
            RMSNorm(args.hidden_size, args.rms_norm_eps);
        register_module("self_attn", self_attn);
        register_module("mlp", mlp);
        register_module("input_layernorm", input_layernorm);
        register_module("post_attention_layernorm", post_attention_layernorm);
    }
    array forward(array x)
    {
        array r = self_attn.forward(input_layernorm.forward(x));
        array h = x + r;
        r = mlp.forward(post_attention_layernorm.forward(h));
        array out = h + r;
        return out;
    }
};
class Phi3Model : public nn::Module
{
public:
    struct PhiModelConfig args;
    int vocab_size, num_hidden_layers;
    Embedding embed_tokens;
    std::vector<TransformerBlock> layers{};
    RMSNorm norm;

    Phi3Model() = default;
    Phi3Model(struct PhiModelConfig _args)
    {
        args = _args;
        vocab_size = args.vocab_size;
        num_hidden_layers = args.num_hidden_layers;
        embed_tokens = Embedding(args.vocab_size, args.hidden_size);
        for (size_t i = 0; i < args.num_hidden_layers; i++)
        {
            layers.push_back(TransformerBlock(args));
        }
        norm = RMSNorm(args.hidden_size, args.rms_norm_eps);

        register_module("embed_tokens", embed_tokens);
        register_layer("layers", layers);
        register_module("norm", norm);
    }
    array forward(array x)
    {
        array h = embed_tokens.forward(x);
        for (auto &l : layers)
        {
            h = l.forward(x);
        }
        return norm.forward(h);
    }
};

class Model : public nn::Module
{
public:
    std::string model_type;
    struct PhiModelConfig args;
    Phi3Model model;
    LinearLayer lm_head;

    Model() = default;
    Model(struct PhiModelConfig _args)
    {
        args = _args;
        model_type = args.model_type;
        model = Phi3Model(args);
        lm_head = LinearLayer(args.hidden_size, args.vocab_size, false);
    }

    array forward(array x)
    {
        array out = model.forward(x);
        eval(out);
        return lm_head.forward(out);
    }

    int head_dim()
    {
        return int(args.hidden_size / args.num_attention_heads);
    }
    int n_kv_heads()
    {
        return args.num_key_value_heads;
    }
};
