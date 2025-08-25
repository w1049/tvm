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

#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/target/target.h>
#include <tvm/ffi/container/array.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace backend {

using meta_schedule::ExtractedTask;

/*!
 * \brief Generate execution sequence of a DNN represented by task IDs
 * \note This function traverses the IRModule and records the execution order
 * of tasks identified by their task names. It validates that all tasks found
 * in the module exist in the provided task array.
 */
class ExecutionSequenceExtractor : public ExprVisitor {
 public:
  static Array<Integer> GetExecutionSequence(IRModule mod, Array<ExtractedTask> tasks) {
    ExecutionSequenceExtractor extractor(std::move(tasks));

    // Build task name to ID mapping
    extractor.BuildTaskMap();

    // Traverse all Relax functions in the module
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<FunctionNode>()) {
        extractor(GetRef<Function>(func));
      }
    }

    return extractor.execution_sequence_;
  }

 private:
  explicit ExecutionSequenceExtractor(Array<ExtractedTask> tasks) : tasks_(std::move(tasks)) {}

  void BuildTaskMap() {
    for (size_t i = 0; i < tasks_.size(); ++i) {
      task_name_to_id_[tasks_[i]->task_name] = static_cast<int>(i);
    }
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    if (!call->op.same_as(call_tir_op)) {
      // Skip non-TIR calls
      return;
    }

    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    const String& task_name = global_var->name_hint;

    // Check if the task exists in the provided task array
    auto it = task_name_to_id_.find(task_name);
    if (it == task_name_to_id_.end()) {
      LOG(FATAL) << "Task '" << task_name
                 << "' found in module does not exist in provided task array";
    }

    // Add task ID to execution sequence
    execution_sequence_.push_back(Integer(it->second));

    // Continue traversal to maintain execution order
    ExprVisitor::VisitExpr_(call);
  }

  Array<ExtractedTask> tasks_;
  std::unordered_map<String, int> task_name_to_id_;
  Array<Integer> execution_sequence_;
};

/*!
 * \brief Public interface for generating execution sequence
 * \param mod The IRModule to analyze
 * \param tasks Array of ExtractedTask objects with known task IDs
 * \return Array of task IDs representing execution order
 * \throws Fatal error if module contains task not found in task array
 */
Array<Integer> GetExecutionSequence(IRModule mod, Array<ExtractedTask> tasks) {
  return ExecutionSequenceExtractor::GetExecutionSequence(std::move(mod), std::move(tasks));
}

TVM_FFI_REGISTER_GLOBAL("relax.backend.GetExecutionSequence")
    .set_body_typed([](IRModule mod, Array<ExtractedTask> tasks) {
      return GetExecutionSequence(std::move(mod), std::move(tasks));
    });

}  // namespace backend
}  // namespace relax
}  // namespace tvm
