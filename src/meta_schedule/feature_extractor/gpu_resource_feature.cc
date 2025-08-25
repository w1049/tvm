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
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/meta_schedule/feature_extractor.h>
#include <tvm/tir/transform.h>

#include <memory>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace tir {

namespace utils {
std::vector<int64_t> GetBufferShape(const Buffer& buffer, arith::Analyzer* analyzer);
runtime::NDArray AsNDArray(const std::vector<std::vector<double>>& src, int second_dim_size = -1);
}  // namespace utils

namespace transform {

/*!
 * \brief Create a list of passes that preprocesses the IR for GPU resource feature extraction
 * \return The list of passes created
 */
Sequential PassListForGPUResourceFeature() {
  return Sequential({
      tir::transform::RemoveWeightLayoutRewriteBlock(/*skip_ndarray_rewrite*/ true),
      tir::transform::LowerCrossThreadReduction(),
      tir::transform::LowerInitBlock(),
      tir::transform::PlanAndUpdateBufferAllocationLocation(),
      tir::transform::ConvertBlocksToOpaque(),
      tir::transform::CompactBufferAllocation(),
      tir::transform::Simplify(),
      tir::transform::LowerAutoCopy(),
      tir::transform::UnifyThreadBinding(),
      tir::transform::LowerMatchBuffer(),
      tir::transform::Simplify(),
  });
}

}  // namespace transform

/*! \brief A data structure for collecting GPU resource features */
struct GPUResourceFeatures {
  // 0 for not set; convert to 1 when exporting
  int64_t blockIdx_x = 0;   // Block count in x dimension
  int64_t blockIdx_y = 0;   // Block count in y dimension
  int64_t blockIdx_z = 0;   // Block count in z dimension
  int64_t threadIdx_x = 0;  // Thread count in x dimension
  int64_t threadIdx_y = 0;  // Thread count in y dimension
  int64_t threadIdx_z = 0;  // Thread count in z dimension

  int64_t shared_memory = 0;  // Shared memory usage in bytes

  static constexpr int64_t kCount = 7;

  void Export(std::vector<double>* v) const {
    double vs[] = {static_cast<double>(blockIdx_x ? blockIdx_x : 1),
                   static_cast<double>(blockIdx_y ? blockIdx_y : 1),
                   static_cast<double>(blockIdx_z ? blockIdx_z : 1),
                   static_cast<double>(threadIdx_x ? threadIdx_x : 1),
                   static_cast<double>(threadIdx_y ? threadIdx_y : 1),
                   static_cast<double>(threadIdx_z ? threadIdx_z : 1),
                   static_cast<double>(shared_memory)};
    v->insert(v->end(), std::begin(vs), std::end(vs));
  }
};

/*! \brief The main GPU resource feature extractor */
class GPUResourceFeatureCollector : private StmtVisitor {
 public:
  static GPUResourceFeatures Collect(const IRModule& mod) {
    GPUResourceFeatureCollector collector;
    for (const auto& kv : mod->functions) {
      if (const PrimFuncNode* func = kv.second.as<PrimFuncNode>()) {
        collector(func->body);
      }
    }
    return collector.features_;
  }

 private:
  void VisitStmt_(const ForNode* loop) final {
    if (loop->kind == ForKind::kThreadBinding) {
      std::string thread_tag = loop->thread_binding.value()->thread_tag;
      if (const int64_t* extent = GetLoopIntExtent(loop)) {
        if (thread_tag == "blockIdx.x" && features_.blockIdx_x == 0) {
          features_.blockIdx_x = *extent;
        } else if (thread_tag == "blockIdx.y" && features_.blockIdx_y == 0) {
          features_.blockIdx_y = *extent;
        } else if (thread_tag == "blockIdx.z" && features_.blockIdx_z == 0) {
          features_.blockIdx_z = *extent;
        } else if (thread_tag == "threadIdx.x" && features_.threadIdx_x == 0) {
          features_.threadIdx_x = *extent;
        } else if (thread_tag == "threadIdx.y" && features_.threadIdx_y == 0) {
          features_.threadIdx_y = *extent;
        } else if (thread_tag == "threadIdx.z" && features_.threadIdx_z == 0) {
          features_.threadIdx_z = *extent;
        }
      }
    }
    auto it1 = loop->annotations.find(attr::software_pipeline_stage);
    int multipiler = 1;
    if (it1 != loop->annotations.end()) {
      auto stage = Downcast<Array<Integer>>((*it1).second);
      auto size = stage.size();
      multipiler = stage[size - 1].IntValue() + 1;
    }
    multipiler_ *= multipiler;
    StmtVisitor::VisitStmt_(loop);
    multipiler_ /= multipiler;
  }

  void VisitStmt_(const BlockNode* block) final {
    StmtVisitor::VisitStmt_(block);
    for (const Buffer& buffer : block->alloc_buffers) {
      if (buffer.scope() == "shared") {
        std::vector<int64_t> shape = utils::GetBufferShape(buffer, &analyzer_);
        int64_t numel = 1;
        for (int64_t x : shape) {
          numel *= x;
        }
        features_.shared_memory += numel * buffer->dtype.bytes() * multipiler_;
      }
    }
  }

  arith::Analyzer analyzer_;
  GPUResourceFeatures features_;
  int multipiler_ = 1;
};

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

class GPUResourceFeatureNode : public FeatureExtractorNode {
 public:
  int feature_vector_length;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<GPUResourceFeatureNode>().def_ro(
        "feature_vector_length", &GPUResourceFeatureNode::feature_vector_length);
  }

  void ExtractSingle(IRModule mod, bool is_gpu, std::vector<std::vector<double>>* results) {
    static transform::Sequential passes = tir::transform::PassListForGPUResourceFeature();
    mod = passes(std::move(mod));

    if (!is_gpu) {
      results->resize(1);
      (*results)[0] = std::vector<double>(feature_vector_length, 0.0);
      return;
    }

    tir::GPUResourceFeatures features = tir::GPUResourceFeatureCollector::Collect(mod);
    results->resize(1);
    std::vector<double>& result = (*results)[0];
    result.reserve(feature_vector_length);
    features.Export(&result);
  }

  Array<runtime::NDArray> ExtractFrom(const TuneContext& tune_context,
                                      const Array<MeasureCandidate>& candidates) {
    auto& target_keys = tune_context->target.value()->keys;
    bool is_gpu = std::find(target_keys.begin(), target_keys.end(), "gpu") != target_keys.end();
    std::vector<runtime::NDArray> results;
    results.resize(candidates.size());

    auto f = [this, is_gpu, &candidates, &results](int, int task_id) -> void {
      const auto& candidate = candidates[task_id];
      std::vector<std::vector<double>> features;
      ExtractSingle(DeepCopyIRModule(candidate->sch->mod()), is_gpu, &features);
      results[task_id] = tir::utils::AsNDArray(features, this->feature_vector_length);
    };

    support::parallel_for_dynamic(0, candidates.size(), tune_context->num_threads, f);
    return results;
  }

  static constexpr const char* _type_key = "meta_schedule.GPUResourceFeature";
  TVM_DECLARE_FINAL_OBJECT_INFO(GPUResourceFeatureNode, FeatureExtractorNode);
};

FeatureExtractor FeatureExtractor::GPUResourceFeature() {
  ObjectPtr<GPUResourceFeatureNode> n = make_object<GPUResourceFeatureNode>();
  n->feature_vector_length = tir::GPUResourceFeatures::kCount;
  return FeatureExtractor(n);
}

TVM_FFI_STATIC_INIT_BLOCK({ GPUResourceFeatureNode::RegisterReflection(); });

TVM_REGISTER_NODE_TYPE(GPUResourceFeatureNode);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.FeatureExtractorGPUResourceFeature")
    .set_body_typed(FeatureExtractor::GPUResourceFeature);

}  // namespace meta_schedule
}  // namespace tvm
