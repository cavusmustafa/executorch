/*  Copyright (c) Intel Corporation
 *
 *  Licensed under the BSD License (the "License"); you may not use this file
 *  except in compliance with the License. See the license file found in the
 *  LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <iostream>
#include <memory>

#include <openvino/openvino.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "OpenvinoBackend.h"

namespace executorch {
namespace backends {
namespace openvino {

OpenvinoBackend::OpenvinoBackend() {}

bool OpenvinoBackend::is_available() const {
  try {
    // Create an OpenVINO Core object to verify runtime availability
    ov::Core core;

    // Check if at least one device is available
    auto devices = core.get_available_devices();
    if (!devices.empty()) {
      return true; // OpenVINO is available
    }
  } catch (const std::exception& e) {
    // Log the exception if OpenVINO runtime is not available
    ET_LOG(Error, "OpenVINO is not available: %s", e.what());
  } catch (...) {
    // Handle any unexpected errors
    ET_LOG(
        Error, "OpenVINO availability check failed due to an unknown error.");
  }

  return false; // OpenVINO is not available
}

exr::Result<exr::DelegateHandle*> OpenvinoBackend::init(
    exr::BackendInitContext& context,
    exr::FreeableBuffer* processed,
    exr::ArrayRef<exr::CompileSpec> compile_specs) const {
  ET_LOG(Info, "OpenvinoBackend::init %p", processed->data());

  ov::Core core;
  const char* data_ptr = static_cast<const char*>(processed->data());
  size_t data_size = processed->size();

  // Copy data to a string or vector
  std::string data_string(data_ptr, data_size);

  // Wrap the data in a stream
  std::istringstream compiled_stream(data_string);

  auto device = "CPU";
  // Get the device value, if provided in compile sepcs
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, "device") == 0)
      device = static_cast<char*>(compile_spec.value.buffer);
  }

  // Import the model
  auto compiled_model = core.import_model(compiled_stream, device);

  // The processed data can be freed since the model is compiled
  processed->Free();

  // Allocate an infer request
  std::shared_ptr<ov::InferRequest> infer_request =
      std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());

  // Allocate execution handle
  exr::MemoryAllocator* allocator = context.get_runtime_allocator();
  ExecutionHandle* handle = allocator->allocateInstance<ExecutionHandle>();
  new (handle) ExecutionHandle;
  handle->compiled_model = std::make_shared<ov::CompiledModel>(compiled_model);
  handle->infer_request = infer_request;

  return handle;
}

exr::Error OpenvinoBackend::execute(
    exr::BackendExecutionContext& context,
    exr::DelegateHandle* input_handle,
    exr::EValue** args) const {
  ExecutionHandle* execution_handle = (ExecutionHandle*)input_handle;

  auto infer_request = execution_handle->infer_request;

  size_t num_inputs = infer_request->get_compiled_model().inputs().size();
  size_t num_outputs = infer_request->get_compiled_model().outputs().size();

  // Set inputs
  std::cout << "DEBUG - OpenvinoBackend - num_inputs: " << num_inputs << std::endl;
  for (size_t i = 0; i < num_inputs; i++) {
    std::cout << "DEBUG - OpenvinoBackend - input - A - i: " << i << std::endl;

    if (args[i]->isNone()) {
        std::cout << "DEBUG - Module - forward - A - type: none" << std::endl;
    } else if (args[i]->isInt()) {
        std::cout << "DEBUG - Module - forward - A - type: int, val: " << args[i]->toInt() << std::endl;
    } else if (args[i]->isDouble()) {
        std::cout << "DEBUG - Module - forward - A - type: double" << std::endl;
    } else if (args[i]->isBool()) {
        std::cout << "DEBUG - Module - forward - A - type: bool" << std::endl;
    } else if (args[i]->isScalar()) {
        std::cout << "DEBUG - Module - forward - A - type: scalar" << std::endl;
    } else if (args[i]->isTensor()) {
        std::cout << "DEBUG - Module - forward - A - type: tensor, shape: [";
        for (int j=0; j<args[i]->toTensor().dim(); j++) {
            std::cout << args[i]->toTensor().size(j) << ", ";
        }
        std::cout << "]" << std::endl;
    } else if (args[i]->isString()) {
        std::cout << "DEBUG - Module - forward - A - type: string" << std::endl;
    } else if (args[i]->isIntList()) {
        std::cout << "DEBUG - Module - forward - A - type: int_list" << std::endl;
    } else if (args[i]->isBoolList()) {
        std::cout << "DEBUG - Module - forward - A - type: bool_list" << std::endl;
    } else if (args[i]->isDoubleList()) {
        std::cout << "DEBUG - Module - forward - A - type: double_list" << std::endl;
    } else if (args[i]->isTensorList()) {
        std::cout << "DEBUG - Module - forward - A - type: tensor_list" << std::endl;
    } else if (args[i]->isListOptionalTensor()) {
        std::cout << "DEBUG - Module - forward - A - type: list_optional_tensor" << std::endl;
    } else {
        std::cout << "DEBUG - Module - forward - A - type: no type available" << std::endl;
    }

    if (args[i]->isInt()) {
        //std::cout << "DEBUG - OpenvinoBackend - input - B.1" << std::endl;
        //auto input_tensor = args[i]->toInt();
        //std::cout << "DEBUG - OpenvinoBackend - input - B.2" << std::endl;
        //ov::Shape input_shape(
        //    input_tensor.sizes().begin(), input_tensor.sizes().end());

        //std::cout << "DEBUG - OpenvinoBackend - input - B.3" << std::endl;
        // Convert input tensor to OpenVINO tensor
        //std::cout << "DEBUG - OpenvinoBackend - input - B.4" << std::endl;
        //int64_t val = args[i]->toInt();
        //int64_t val = i;
        int64_t *val = &(args[i]->payload.copyable_union.as_int);
        //std::cout << "DEBUG - OpenvinoBackend - input - B.5 - val: " << val << std::endl;
        //ov::Tensor ov_input_tensor(ov::element::i64, ov::Shape{}, &val);
        //std::vector<int64_t> val = {args[i]->toInt()};
        //ov::Tensor ov_input_tensor(ov::element::i64, ov::Shape{1}, &val);
        ov::Tensor ov_input_tensor(ov::element::i64, ov::Shape{1}, val);
        std::cout << "DEBUG - OpenvinoBackend - input - B.6 - val: " << ((int64_t*)(ov_input_tensor.data<int64_t>()))[0] << ", byte_size: " << ov_input_tensor.get_byte_size() << std::endl;

        infer_request->set_input_tensor(i, ov_input_tensor);
        //std::cout << "DEBUG - OpenvinoBackend - input - B.7" << std::endl;
    } else {
        //std::cout << "DEBUG - OpenvinoBackend - input - C.1" << std::endl;
        auto input_tensor = args[i]->toTensor();
        //std::cout << "DEBUG - OpenvinoBackend - input - C.2" << std::endl;
        ov::Shape input_shape(
            input_tensor.sizes().begin(), input_tensor.sizes().end());

        //std::cout << "DEBUG - OpenvinoBackend - input - C.3" << std::endl;
        // Convert input tensor to OpenVINO tensor
        ov::element::Type ov_type =
            convert_to_openvino_type(input_tensor.scalar_type());
        //std::cout << "DEBUG - OpenvinoBackend - input - C.4" << std::endl;
        ov::Tensor ov_input_tensor(
            ov_type, input_shape, input_tensor.mutable_data_ptr());
        //std::cout << "DEBUG - OpenvinoBackend - input - C.5" << std::endl;

        infer_request->set_input_tensor(i, ov_input_tensor);
        //std::cout << "DEBUG - OpenvinoBackend - input - C.6" << std::endl;
    }
  }

  // Set outputs
  std::cout << "DEBUG - OpenvinoBackend - num_outputs: " << num_outputs << std::endl;
  for (size_t i = 0; i < num_outputs; i++) {
    //args[num_inputs + i]->toTensor().unsafeGetTensorImpl()->set_size(1,1);
    std::cout << "DEBUG - OpenvinoBackend output - i: " << i << " - type: tensor, shape: [";
    for (int j=0; j<args[num_inputs + i]->toTensor().dim(); j++) {
        std::cout << args[num_inputs + i]->toTensor().size(j) << ", ";
    }
    std::cout << "]" << std::endl; 
    auto output_tensor = args[num_inputs + i]->toTensor();
    ov::Shape output_shape(
        output_tensor.sizes().begin(), output_tensor.sizes().end());

    // Convert input tensor to OpenVINO tensor
    ov::element::Type ov_type =
        convert_to_openvino_type(output_tensor.scalar_type());
    ov::Tensor ov_output_tensor(
        ov_type, output_shape, output_tensor.mutable_data_ptr());

    infer_request->set_output_tensor(i, ov_output_tensor);
  }

  // Execute the inference
  infer_request->infer();
  //auto out_t = infer_request->get_output_tensor(0);
  //std::cout << "DEBUG - OpenvinoBackend output - after infer tensor - shape: " << out_t.get_shape() << std::endl;
  //for (int j=0; j<args[num_inputs + i]->toTensor().dim(); j++) {
  //    std::cout << args[num_inputs + i]->toTensor().size(j) << ", ";
  //}
  //std::cout << "]" << std::endl;

  //std::cout << "DEBUG - OpenvinoBackend - DD" << std::endl;
  return exr::Error::Ok;
}

void OpenvinoBackend::destroy(exr::DelegateHandle* handle) const {
  if (!handle) {
    ET_LOG(Info, "Attempted to destroy a null handle.");
    return;
  }

  // Cast the handle to the appropriate type
  ExecutionHandle* execution_handle = static_cast<ExecutionHandle*>(handle);

  // Clean up resources
  if (execution_handle->infer_request) {
    execution_handle->infer_request.reset(); // Release the infer request
    ET_LOG(Info, "Infer request successfully destroyed.");
  }

  if (execution_handle->compiled_model) {
    execution_handle->compiled_model.reset(); // Release the compiled model
    ET_LOG(Info, "Compiled model successfully destroyed.");
  }

  ET_LOG(Info, "Delegate handle destroyed successfully.");
}

ov::element::Type OpenvinoBackend::convert_to_openvino_type(
    exa::ScalarType scalar_type) const {
  switch (scalar_type) {
    case exa::ScalarType::Float:
      return ov::element::f32;
    case exa::ScalarType::Int:
      return ov::element::i32;
    case exa::ScalarType::Char:
      return ov::element::i8;
    case exa::ScalarType::Long:
      return ov::element::i64;
    case exa::ScalarType::Bool:
      return ov::element::boolean;
    default:
      throw std::runtime_error("Unsupported scalar type");
  }
}

} // namespace openvino
} // namespace backends
} // namespace executorch

namespace {
auto backend = executorch::backends::openvino::OpenvinoBackend();
executorch::runtime::Backend backend_id{"OpenvinoBackend", &backend};
static auto registered = executorch::runtime::register_backend(backend_id);
} // namespace
