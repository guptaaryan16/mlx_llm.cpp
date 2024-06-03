# MLX_LLM.cpp

The project is an attempt to create the high level C++ Frontend API for MLX Framework, adding LLM models similar to llama.cpp. This is done to use the useful array API developed by the contributions of the Apple team and the MLX community to bring NN on M series MacBook GPUs.   

### Motivation 
- Developed in an effort to use the amazing array API of the MLX framework on all native C/C++,
  including WasmEdge 
- Highly inspired by PyTorch Cpp frontend API, this will help in usage of API in low latency applications and shared threads with mutex(maybe in the future)

### Requirements:
- M-Series processor Apple device
- MLX installed (to install MLX, visit the [docs](https://ml-explore.github.io/mlx/build/html/install.html))
- Apple Clang 15.0 (used in the development) 
- Cmake (3.24 and higher)
- XCode Tool Chain

### Installation 
1. Build and Install the mlx library from the [documentation](https://ml-explore.github.io/mlx/build/html/install.html)
2. Git clone the repository.
   ```
   git clone https://github.com/guptaaryan16/mlx_llm.cpp
   ```
3. Run the commands 
   ```
   cd mlx_llm.cpp
   cmake .  
   make 
   ./MLX_llm_output test_nn.safetensors
   ```
   The above command runs the test file for the mlx_llm API. More API for the models is expected in future.

#### Note: 
Step 3 would build the library. For specific steps, refer to the MLX and CMAKE documentation.

### What works 
- Able to write Module(very similar to the python API) for the project.  
- Able to load weights using both `.safetensors` and `.gguf` formats
- Able to do steady inference on smaller neural networks

### TODO:
- Complete Phi3 and LLAMA3 examples
- Correct the behavior of loading and saving tensors 
- Build more Layers and work with the API of Primitives. For more info, read https://ml-explore.github.io/mlx/build/html/dev/extensions.html

### Usage with WasmEdge 
WIP

### LICENSE 
The above code is licensed under MIT LICENSE. 

### Thanks
This API is developed as part of LFX mentorship program, which meant integrating the MLX framework with the WasmEdge Runtime Environment. Thus, I really would like to thank my mentors and the LFX team for supporting the development of the project.

Also I would like to thank the MLX maintainers and community for the support.

Read more about my experience and the blog post here:
