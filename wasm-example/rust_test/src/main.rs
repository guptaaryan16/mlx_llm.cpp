use std::env;
use rand::Rng;

pub fn main() {
    let args: Vec<String> = env::args().collect();
    let model_bin_name: &str = &args[1];
    println!("model_bin_name {}", model_bin_name);

    let graph = wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::MLX,
        wasi_nn::ExecutionTarget::CPU,
    )
    .build_from_cache(&model_bin_name)
    .expect("Failed to load the model");

    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().expect("Failed to init context");
    println!("Created wasi-nn execution context with ID: {:?}", context);

    // Load a tensor that precisely matches the graph input tensor dimensions
    // Graph expects  dimensional tensor
    let tensor_data: Vec<f32> = (0..784)
                                .map(|_| rand::thread_rng()
                                .gen_range(0.0..1.0))
                                .collect();

    println!("Read input tensor, size in bytes: {}", tensor_data.len());

    context
        .set_input(
            0,
            wasi_nn::TensorType::F32,
            // Input
            &[1, 784],
            &tensor_data,
        )
        .unwrap();

    // // Execute the inference.
    context.compute().unwrap();
    println!("Executed graph inference");

    let mut output_buffer = vec![0f32; 10];
    context.get_output(0, &mut output_buffer).unwrap();

    println!("{:?}", output_buffer);

}

