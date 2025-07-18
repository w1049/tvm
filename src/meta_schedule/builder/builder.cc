/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/******** Constructors ********/

BuilderInput::BuilderInput(IRModule mod, Target target,
                           Optional<Map<String, runtime::NDArray>> params) {
  ObjectPtr<BuilderInputNode> n = make_object<BuilderInputNode>();
  n->mod = std::move(mod);
  n->target = std::move(target);
  n->params = std::move(params);
  data_ = std::move(n);
}

BuilderResult::BuilderResult(Optional<String> artifact_path, Optional<Array<Integer>> extra_info,
                             Optional<String> error_msg) {
  ObjectPtr<BuilderResultNode> n = make_object<BuilderResultNode>();
  n->artifact_path = std::move(artifact_path);
  n->extra_info = std::move(extra_info);
  n->error_msg = std::move(error_msg);
  data_ = std::move(n);
}

Builder Builder::PyBuilder(BuilderNode::FBuild f_build) {
  ObjectPtr<PyBuilderNode> n = make_object<PyBuilderNode>();
  n->f_build = std::move(f_build);
  return Builder(std::move(n));
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK({
  BuilderInputNode::RegisterReflection();
  BuilderResultNode::RegisterReflection();
  PyBuilderNode::RegisterReflection();
});

TVM_REGISTER_NODE_TYPE(BuilderInputNode);
TVM_REGISTER_NODE_TYPE(BuilderResultNode);
TVM_REGISTER_OBJECT_TYPE(BuilderNode);
TVM_REGISTER_NODE_TYPE(PyBuilderNode);

TVM_FFI_REGISTER_GLOBAL("meta_schedule.BuilderInput")
    .set_body_typed([](IRModule mod, Target target,
                       Optional<Map<String, runtime::NDArray>> params) -> BuilderInput {
      return BuilderInput(mod, target, params);
    });

TVM_FFI_REGISTER_GLOBAL("meta_schedule.BuilderResult")
    .set_body_typed([](Optional<String> artifact_path, Optional<Array<Integer>> extra_info,
                       Optional<String> error_msg) -> BuilderResult {
      return BuilderResult(artifact_path, extra_info, error_msg);
    });

TVM_FFI_REGISTER_GLOBAL("meta_schedule.BuilderBuild").set_body_method(&BuilderNode::Build);

TVM_FFI_REGISTER_GLOBAL("meta_schedule.BuilderPyBuilder").set_body_typed(Builder::PyBuilder);

}  // namespace meta_schedule
}  // namespace tvm
