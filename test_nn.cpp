#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "mlx/mlx.h"
#include "mlx_llm/llm.h"

int main() {
  array input = random::uniform({1, 784});
  MyModel model = MyModel();
  auto res = model.forward(input);
  std::cout << res << "\n" << res.shape();
  model.print_parameters();
  model.load_weights(
      "/Users/guptaaryan16/Desktop/OSS/mlx/examples/cpp/mymodel.safetensors");
  auto res_ = model.forward(input);
  model.print_parameters();
  std::cout << res_ << "\n" << res_.shape();
  return 0;
}