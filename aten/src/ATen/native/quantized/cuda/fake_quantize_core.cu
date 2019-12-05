#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cmath>

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
namespace at {
namespace native {
void fake_quantize_slice_cuda(
    Tensor& output,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t rounding_method) {
  float inv_scale = 1.0f / scale;
  switch(rounding_method) {
  case 0:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::nearbyint(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
    break;
  case 1:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::round(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
    break;
  case 2:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::floor(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
    break;
  case 3:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::ceil(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
    break;
  case 4:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::trunc(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
    break;
  default:
    at::cuda::CUDA_tensor_apply2<float, float>(
      input, output, [=] __device__(const float& input_val, float& result_val) {
        result_val = (fminf(
                          quant_max,
                          fmaxf(
                              quant_min,
                              static_cast<int64_t>(std::nearbyint(
                                  input_val * inv_scale + zero_point)))) -
                      zero_point) *
            scale;
      });
  }
}

void fake_quantize_grad_slice_cuda(
    Tensor& input_grad,
    const Tensor& input,
    const Tensor& output_grad,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t rounding_method) {
  float inv_scale = 1.0f / scale;
  switch(rounding_method) {
  case 0:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    break;
  case 1:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::round(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    break;
  case 2:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::floor(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    break;
  case 3:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::ceil(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    break;
  case 4:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::trunc(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
    break;
  default:
    at::cuda::CUDA_tensor_apply3<float, float, float>(
      output_grad,
      input,
      input_grad,
      [=] __device__(const float& dy, const float& x, float& dx) {
        int64_t Xq = std::nearbyint(x * inv_scale + zero_point);
        dx = (Xq >= quant_min && Xq <= quant_max) * dy;
      });
  }
}

} // namespace native
} // namespace at
