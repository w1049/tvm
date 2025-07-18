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

#include "./utils.h"
namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK({
  BlockRVNode::RegisterReflection();
  LoopRVNode::RegisterReflection();
});

/**************** Constructor ****************/

BlockRV::BlockRV() { this->data_ = make_object<BlockRVNode>(); }

LoopRV::LoopRV() { this->data_ = make_object<LoopRVNode>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleNode::GetSRef(const StmtNode* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_OBJECT_TYPE(ScheduleNode);

TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetMod")  //
    .set_body_method(&ScheduleNode::mod);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetState")  //
    .set_body_method(&ScheduleNode::state);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetTrace")  //
    .set_body_method(&ScheduleNode::trace);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetFuncWorkingOn")  //
    .set_body_method(&ScheduleNode::func_working_on);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleCopy")  //
    .set_body_method(&ScheduleNode::Copy);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSeed")  //
    .set_body_method(&ScheduleNode::Seed);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleForkSeed")  //
    .set_body_method(&ScheduleNode::ForkSeed);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleWorkOn")  //
    .set_body_method(&ScheduleNode::WorkOn);

/**************** (FFI) Constructor ****************/

TVM_FFI_REGISTER_GLOBAL("tir.schedule.BlockRV").set_body_typed([]() { return BlockRV(); });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.LoopRV").set_body_typed([]() { return LoopRV(); });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ConcreteSchedule")
    .set_body_typed([](IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                       int debug_mask, int error_render_level, bool enable_check) -> Schedule {
      return Schedule::Concrete(mod, debug_mask, seed,
                                static_cast<ScheduleErrorRenderLevel>(error_render_level),
                                enable_check);
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.TracedSchedule")
    .set_body_typed([](IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                       int debug_mask, int error_render_level, bool enable_check) -> Schedule {
      return Schedule::Traced(mod, seed, debug_mask,
                              static_cast<ScheduleErrorRenderLevel>(error_render_level),
                              enable_check);
    });

/******** (FFI) Lookup random variables ********/

TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGet")
    .set_body_typed([](Schedule self, ObjectRef obj) -> ObjectRef {
      if (auto loop_rv = obj.as<LoopRV>()) {
        return self->Get(loop_rv.value());
      }
      if (auto block_rv = obj.as<BlockRV>()) {
        return self->Get(block_rv.value());
      }
      if (auto expr_rv = obj.as<ExprRV>()) {
        return self->Get(expr_rv.value());
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
      throw;
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetSRef")
    .set_body_typed([](Schedule self, ObjectRef obj) -> Optional<ObjectRef> {
      if (auto loop_rv = obj.as<LoopRV>()) {
        return self->GetSRef(loop_rv.value());
      }
      if (auto block_rv = obj.as<BlockRV>()) {
        return self->GetSRef(block_rv.value());
      }
      if (auto stmt = obj.as<Stmt>()) {
        return self->GetSRef(stmt.value());
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleRemoveRV")
    .set_body_typed([](Schedule self, ObjectRef obj) -> void {
      if (auto loop_rv = obj.as<LoopRV>()) {
        return self->RemoveRV(loop_rv.value());
      }
      if (auto block_rv = obj.as<BlockRV>()) {
        return self->RemoveRV(block_rv.value());
      }
      if (auto expr_rv = obj.as<ExprRV>()) {
        return self->RemoveRV(expr_rv.value());
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });

/******** (FFI) Sampling ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSampleCategorical")
    .set_body_method(&ScheduleNode::SampleCategorical);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSamplePerfectTile")
    .set_body_method(&ScheduleNode::SamplePerfectTile);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSamplePartitionedTile")
    .set_body_method(&ScheduleNode::SamplePartitionedTile);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSampleComputeLocation")
    .set_body_method(&ScheduleNode::SampleComputeLocation);
/******** (FFI) Get blocks & loops ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetBlock").set_body_method(&ScheduleNode::GetBlock);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetLoops").set_body_method(&ScheduleNode::GetLoops);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetChildBlocks")
    .set_body_typed([](Schedule self, ObjectRef rv) {
      if (auto block_rv = rv.as<BlockRV>()) {
        return self->GetChildBlocks(block_rv.value());
      }
      if (auto loop_rv = rv.as<LoopRV>()) {
        return self->GetChildBlocks(loop_rv.value());
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
      throw;
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetProducers")
    .set_body_method(&ScheduleNode::GetProducers);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetConsumers")
    .set_body_method(&ScheduleNode::GetConsumers);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleGetOutputBlocks")
    .set_body_method(&ScheduleNode::GetOutputBlocks);
/******** (FFI) Transform loops ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleMerge").set_body_method(&ScheduleNode::Merge);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleFuse").set_body_method(&ScheduleNode::Fuse);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSplit").set_body_method(&ScheduleNode::Split);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleLoopPartition")
    .set_body_method(&ScheduleNode::LoopPartition);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReorder").set_body_method(&ScheduleNode::Reorder);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReorderBlockIterVar")
    .set_body_method(&ScheduleNode::ReorderBlockIterVar);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleAddUnitLoop")
    .set_body_typed([](Schedule self, ObjectRef rv) -> LoopRV {
      if (auto loop_rv = rv.as<LoopRV>()) {
        return self->AddUnitLoop(loop_rv.value());
      } else if (auto block_rv = rv.as<BlockRV>()) {
        return self->AddUnitLoop(block_rv.value());
      } else {
        LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                   << ". Its value is: " << rv;
        throw;
      }
    });
/******** (FFI) Manipulate ForKind ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleParallel").set_body_method(&ScheduleNode::Parallel);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleVectorize").set_body_method(&ScheduleNode::Vectorize);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleBind").set_body_method(&ScheduleNode::Bind);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleUnroll").set_body_method(&ScheduleNode::Unroll);
/******** (FFI) Insert cache stages ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleCacheRead").set_body_method(&ScheduleNode::CacheRead);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleCacheWrite")
    .set_body_method(&ScheduleNode::CacheWrite);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReindexCacheRead")
    .set_body_method(&ScheduleNode::ReindexCacheRead);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReindexCacheWrite")
    .set_body_method(&ScheduleNode::ReindexCacheWrite);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleCacheInplace")
    .set_body_method(&ScheduleNode::CacheInplace);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleCacheIndex")
    .set_body_method(&ScheduleNode::CacheIndex);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReIndex")
    .set_body_typed([](Schedule self, const BlockRV& block_rv, int buffer_index,
                       int buffer_index_type) {
      return self->ReIndex(block_rv, buffer_index, static_cast<BufferIndexType>(buffer_index_type));
    });
/******** (FFI) Data movement ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReadAt").set_body_method(&ScheduleNode::ReadAt);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleWriteAt").set_body_method(&ScheduleNode::WriteAt);
/******** (FFI) Compute location ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleComputeAt").set_body_method(&ScheduleNode::ComputeAt);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeAt")
    .set_body_method(&ScheduleNode::ReverseComputeAt);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleComputeInline")
    .set_body_method(&ScheduleNode::ComputeInline);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeInline")
    .set_body_method(&ScheduleNode::ReverseComputeInline);
/******** (FFI) Reduction ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposeReduction")
    .set_body_method(&ScheduleNode::DecomposeReduction);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleRFactor").set_body_method(&ScheduleNode::RFactor);
/******** (FFI) Block annotation ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleStorageAlign")
    .set_body_method(&ScheduleNode::StorageAlign);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSetScope").set_body_method(&ScheduleNode::SetScope);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleUnsafeSetDType")
    .set_body_method(&ScheduleNode::UnsafeSetDType);
/******** (FFI) Blockize & Tensorize ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleBlockize")
    .set_body_typed([](Schedule self, ObjectRef target, bool preserve_unit_iters) {
      if (auto loop_rv = target.as<LoopRV>()) {
        return self->Blockize(loop_rv.value(), preserve_unit_iters);
      } else if (auto blocks = target.as<Array<BlockRV>>()) {
        return self->Blockize(blocks.value(), preserve_unit_iters);
      }
      LOG(FATAL) << "Unsupported target type: " << target->GetTypeKey();
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleTensorize")
    .set_body_typed([](Schedule self, ObjectRef rv, String intrin, bool preserve_unit_iters) {
      if (auto block_rv = rv.as<BlockRV>()) {
        self->Tensorize(block_rv.value(), intrin, preserve_unit_iters);
      } else if (auto loop_rv = rv.as<LoopRV>()) {
        self->Tensorize(loop_rv.value(), intrin, preserve_unit_iters);
      } else {
        LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                   << ". Its value is: " << rv;
      }
    });

/******** (FFI) Annotation ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleAnnotate")
    .set_body_typed([](Schedule self, ObjectRef rv, const String& ann_key, const Any& ann_val) {
      if (auto block_rv = rv.as<BlockRV>()) {
        return self->Annotate(block_rv.value(), ann_key, ann_val);
      }
      if (auto loop_rv = rv.as<LoopRV>()) {
        return self->Annotate(loop_rv.value(), ann_key, ann_val);
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
      throw;
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleUnannotate")
    .set_body_typed([](Schedule self, ObjectRef rv, const String& ann_key) {
      if (auto block_rv = rv.as<BlockRV>()) {
        return self->Unannotate(block_rv.value(), ann_key);
      }
      if (auto loop_rv = rv.as<LoopRV>()) {
        return self->Unannotate(loop_rv.value(), ann_key);
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << rv->GetTypeKey()
                 << ". Its value is: " << rv;
      throw;
    });

/******** (FFI) Layout transformation ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleTransformLayout")
    .set_body_typed([](Schedule self, const BlockRV& block_rv, int buffer_index,
                       int buffer_index_type, const IndexMap& index_map,
                       const Optional<IndexMap>& pad_value, bool assume_injective_transform) {
      return self->TransformLayout(block_rv, buffer_index,
                                   static_cast<BufferIndexType>(buffer_index_type), index_map,
                                   pad_value, assume_injective_transform);
    });
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleTransformBlockLayout")
    .set_body_method(&ScheduleNode::TransformBlockLayout);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleSetAxisSeparator")
    .set_body_typed([](Schedule self, const BlockRV& block_rv, int buffer_index,
                       int buffer_index_type, const Array<IntImm>& axis_separators) {
      return self->SetAxisSeparator(
          block_rv, buffer_index, static_cast<BufferIndexType>(buffer_index_type), axis_separators);
    });

/******** (FFI) Padding decomposition ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposePadding")
    .set_body_method(&ScheduleNode::DecomposePadding);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.SchedulePadEinsum").set_body_method(&ScheduleNode::PadEinsum);
/******** (FFI) Buffer transformation ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleRollingBuffer")
    .set_body_method(&ScheduleNode::RollingBuffer);
/******** (FFI) Misc ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleEnterPostproc")
    .set_body_method(&ScheduleNode::EnterPostproc);
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleUnsafeHideBufferAccess")
    .set_body_method(&ScheduleNode::UnsafeHideBufferAccess);
/******** (FFI) Annotate buffer access ********/
TVM_FFI_REGISTER_GLOBAL("tir.schedule.ScheduleAnnotateBufferAccess")
    .set_body_typed([](Schedule self, const BlockRV& block_rv, int buffer_index,
                       int buffer_index_type, const IndexMap& index_map) {
      return self->AnnotateBufferAccess(block_rv, buffer_index,
                                        static_cast<BufferIndexType>(buffer_index_type), index_map);
    });

}  // namespace tir
}  // namespace tvm
