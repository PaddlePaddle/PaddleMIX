// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"
#include <vector>

using paddle::Tensor;

namespace paddle {
namespace experimental {


PADDLE_API void softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad);

}
} // namespace paddle



std::vector<Tensor> SoftmaxBwd(const Tensor& grad_output,
                  const Tensor& output,
                  int axis){
    Tensor res;
    paddle::experimental::softmax_grad(output, grad_output, axis, &res);
    return {res};
}


PD_BUILD_OP(softmax_bwd)
    .Inputs({"grad_output", "output"})
    .Outputs({"x_grad"})
    .Attrs({"axis: int"})
    .SetKernelFn(PD_KERNEL(SoftmaxBwd));
