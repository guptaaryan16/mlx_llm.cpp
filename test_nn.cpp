#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "mlx_llm/llm.cpp"

using namespace mlx::core;

int main() {
  array input = random::uniform({1, 784});
  TestModel test_model = TestModel();
  auto res = test_model.forward(input);
  std::cout << res << "\n" << res.shape();
  test_model.print_parameters();
  test_model.load_weights(
      "/Users/guptaaryan16/Desktop/OSS/mlx/examples/cpp/mymodel.safetensors");
  auto res_ = test_model.forward(input);
  test_model.print_parameters();
  std::cout << res_ << "\n" << res_.shape();
  return 0;
}