#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "mlx_llm/llm.cpp"

using namespace mlx::core;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <weights_path>\n";
    return 1;
  }

  // To specify the weights path in absolute as used by MLX API
  std::string weights_path = argv[1];
  if (weights_path[0] != '/') {
    char absolute_path[PATH_MAX];
    if (realpath(weights_path.c_str(), absolute_path) == nullptr) {
      std::cerr << "Failed to convert relative path to absolute path\n";
      return 1;
    }
    weights_path = absolute_path;
  }

  array input = random::uniform({1, 784});
  mlx::core::nn::TestModel test_model = mlx::core::nn::TestModel();
  auto res = test_model.forward(input);
  std::cout << res << "\n" << res.shape();
  test_model.print_parameters();
  if (!weights_path.empty()){
    test_model.load_weights(weights_path);
    auto res_ = test_model.forward(input);
    test_model.print_parameters();
    std::cout << res_ << "\n" << res_.shape();
  }
  return 0;
}