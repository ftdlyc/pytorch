#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

/* Core operations for fake-quantization shared between per tensor
 and per-channel fake quant */
namespace at {
namespace native {

/* Fake quantize a tensor, common block for per-channel & per-tensor fake quant
Args:
  output: output tensor.
  input : input tensor.
  sc:  scale to quantize the input tensor to
  zero_point: zero_point
  quant_min: minimum quantized value
  quant_max: maximum quantized value
Returns:
  Fake quantized tensor (double dtype).
*/
void fake_quantize_slice(
    Tensor& output,
    const Tensor& input,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t rounding_method) {
  float inv_scale = 1.0f / sc;
  auto iter = TensorIterator::unary_op(output, input);
  switch(rounding_method) {
  case 0:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::nearbyint(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
    break;
  case 1:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::round(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
    break;
  case 2:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::floor(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
    break;
  case 3:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::ceil(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
    break;
  case 4:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::trunc(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
  default:
    cpu_kernel(iter, [&](float self) -> float {
      return (std::fmin(
                  std::fmax(
                      static_cast<int64_t>(
                          std::nearbyint(self * inv_scale + z_point)),
                      quant_min),
                  quant_max) -
              z_point) *
          sc;
    });
  }
}

void fake_quantize_grad_slice(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float sc,
    int64_t z_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t rounding_method) {
  float inv_scale = 1.0f / sc;
  auto iter = TensorIterator::binary_op(input_grad, input, output_grad);
  switch(rounding_method) {
  case 0:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::nearbyint(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    break;
  case 1:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::round(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    break;
  case 2:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::floor(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    break;
  case 3:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::ceil(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    break;
  case 4:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::trunc(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
    break;
  default:
    cpu_kernel(iter, [&](float x, float dy) -> float {
      int64_t xq = static_cast<int64_t>(std::nearbyint(x * inv_scale + z_point));
      return dy * (xq >= quant_min && xq <= quant_max);
    });
  }
}

} // namespace native
} // namespace at
