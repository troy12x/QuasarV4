#include <torch/extension.h>
#include <vector>
#include <cmath>

// CUDA Kernel for the element-wise LNN update
__global__ void lnn_elementwise_forward_kernel(
    float* __restrict__ h_ptr, // In-place update
    const float* __restrict__ activation_ptr,
    const float* __restrict__ tau_ptr,
    const float dt,
    const float state_clamp_value,
    const float derivative_clamp_value,
    const int total_elements) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_elements) return;

    // Compute the derivative dx/dt
    const float decay_term = -h_ptr[index] / tau_ptr[index];
    const float activation_output = tanhf(activation_ptr[index]);
    float dx_dt = decay_term + activation_output;

    // Clamp derivative
    dx_dt = fmaxf(-derivative_clamp_value, fminf(derivative_clamp_value, dx_dt));

    // Update hidden state using Euler's method
    h_ptr[index] += dt * dx_dt;

    // Clamp hidden state
    h_ptr[index] = fmaxf(-state_clamp_value, fminf(state_clamp_value, h_ptr[index]));
}

// C++ interface function
torch::Tensor lnn_elementwise_forward(
    torch::Tensor h, // In-place
    torch::Tensor activation,
    torch::Tensor tau,
    float dt,
    float state_clamp_value,
    float derivative_clamp_value) {

    TORCH_CHECK(h.is_cuda(), "h must be a CUDA tensor");
    TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
    TORCH_CHECK(tau.is_cuda(), "tau must be a CUDA tensor");

    const int total_elements = h.numel();
    const int threads = 1024;
    const int blocks = (total_elements + threads - 1) / threads;

    lnn_elementwise_forward_kernel<<<blocks, threads>>>(
        h.data_ptr<float>(),
        activation.data_ptr<float>(),
        tau.data_ptr<float>(),
        dt,
        state_clamp_value,
        derivative_clamp_value,
        total_elements
    );

    return h;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lnn_elementwise_forward, "LNN element-wise forward (CUDA)");
}
