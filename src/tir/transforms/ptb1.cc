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

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../support/utils.h"

namespace tvm {
namespace tir {

using support::StartsWith;

constexpr const char* merge_candidate_key = "merge_candidate";

int convert_to_int(const PrimExpr& expr) {
  if (!expr.defined()) {
    return 0;
  }
  auto analyzer = arith::Analyzer();
  auto simp = analyzer.Simplify(expr);
  const int64_t* pint = tir::as_const_int(simp);
  if (pint) {
    return static_cast<int>(*pint);
  }
  return 0;
}

class ThreadInfoCollector : public StmtVisitor {
 public:
  bool has_multiple_block_idx_{false};
  bool has_different_thread_idx_{false};
  int get_block_sum_extent() const { return convert_to_int(block_sum_extent_); }
  int get_thread_max_extent() const { return convert_to_int(thread_max_extent_); }

 private:
  // Visit the AttrStmt nodes to collect thread extent information.
  void VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == merge_candidate_key) {
      in_merge_scope_ = true;
      this->VisitStmt(op->body);
      in_merge_scope_ = false;
    }
    if (in_merge_scope_ && op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (StartsWith(iv->thread_tag, "blockIdx.")) {
        if (!block_sum_extent_.defined()) {
          block_sum_extent_ = op->value;
        } else {
          has_multiple_block_idx_ = true;
          block_sum_extent_ = block_sum_extent_ + op->value;
        }
      } else if (StartsWith(iv->thread_tag, "threadIdx.")) {
        if (!thread_max_extent_.defined()) {
          thread_max_extent_ = op->value;
        } else {
          if (!thread_max_extent_.same_as(op->value)) {
            has_different_thread_idx_ = true;
            thread_max_extent_ = tvm::max(thread_max_extent_, op->value);
          }
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  PrimExpr block_sum_extent_{nullptr};
  PrimExpr thread_max_extent_{nullptr};
  bool in_merge_scope_{false};
};

class LaunchThreadMerger : public StmtExprMutator {
 public:
  explicit LaunchThreadMerger(const ThreadInfoCollector& collector) : collector_(collector) {}

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == merge_candidate_key) {
      in_merge_scope_ = true;
      Stmt body = this->VisitStmt(op->body);
      in_merge_scope_ = false;
      return body;
    }
    if (in_merge_scope_ && op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (StartsWith(iv->thread_tag, "blockIdx.")) {
        if (!collector_.has_multiple_block_idx_) {
          // If there is only one blockIdx, we can keep it as is.
          return StmtExprMutator::VisitStmt_(op);
        }
        // If there are multiple blockIdx, we need to merge them.
        // In VisitExpr, subtract the current extent from the blockIdx variable.

        // All blockIdx variables should be merged into a single one.
        if (!block_idx_var_.defined()) {
          // Do all range start at 0?
          block_idx_var_ = IterVar(Range::FromMinExtent(0, collector_.get_block_sum_extent()),
                                   iv->var, iv->iter_type, iv->thread_tag, iv->span);
        }

        auto body = this->VisitStmt(op->body);
        auto start_extent = current_extent_;
        current_extent_ += convert_to_int(op->value);
        return AttrStmt(
            block_idx_var_, op->attr_key, collector_.get_block_sum_extent(),
            IfThenElse(And(GE(block_idx_var_, start_extent), LT(block_idx_var_, current_extent_)),
                       body),
            op->span);
      } else if (StartsWith(iv->thread_tag, "threadIdx.")) {
        if (!collector_.has_different_thread_idx_) {
          // If all threadIdx variables have the same extent, we can keep it as is.
          return StmtExprMutator::VisitStmt_(op);
        }
        // "If-then-else" has been placed by InjectDivergentThreadSync,
        // we need to adjust the body to use the max_thread_extent.
        auto body = this->VisitStmt(op->body);
        // Create a new IterVar with the max_thread_extent.
        if (!thread_idx_var_.defined()) {
          thread_idx_var_ = IterVar(Range::FromMinExtent(0, collector_.get_thread_max_extent()),
                                    iv->var, iv->iter_type, iv->thread_tag, iv->span);
        }
        return AttrStmt(thread_idx_var_, op->attr_key, collector_.get_thread_max_extent(), body,
                        op->span);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    // in my use case, top level is SeqStmt { AttrStmt, AttrStmt }
    std::vector<size_t> merge_candidate_indices;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      if (const auto* attr_stmt = op->seq[i].as<AttrStmtNode>()) {
        if (attr_stmt->attr_key == merge_candidate_key) {
          merge_candidate_indices.push_back(i);
        }
      }
    }
    if (merge_candidate_indices.empty()) {
      // No merge candidates found, just visit the statements.
      return StmtExprMutator::VisitStmt_(op);
    }

    // If there are multiple merge candidates, we need to merge them.
    Array<Stmt> new_seq;
    // before merge start
    for (size_t i = 0; i < merge_candidate_indices[0]; ++i) {
      new_seq.push_back(this->VisitStmt(op->seq[i]));
    }

    Array<Stmt> merged_attrs_body;
    AttrStmt first_attr_stmt;
    for (auto i : merge_candidate_indices) {
      auto merge_attr_stmt = op->seq[i].as<AttrStmtNode>();
      ICHECK(merge_attr_stmt) << "Expected AttrStmtNode, but got: " << op->seq[i];
      ICHECK(merge_attr_stmt->attr_key == merge_candidate_key)
          << "Expected attr_key to be '" << merge_candidate_key
          << "', but got: " << merge_attr_stmt->attr_key;
      auto block_idx_stmt = merge_attr_stmt->body.as<AttrStmtNode>();
      ICHECK(block_idx_stmt) << "Expected body to be an AttrStmtNode, but got: "
                             << merge_attr_stmt->body;
      ICHECK(block_idx_stmt->attr_key == tir::attr::thread_extent &&
             StartsWith(Downcast<IterVar>(block_idx_stmt->node)->thread_tag, "blockIdx."))
          << "Expected thread extent attribute for blockIdx, but got: " << block_idx_stmt->attr_key;
      Stmt stmt = this->VisitStmt(op->seq[i]);
      auto attr_stmt = stmt.as<AttrStmtNode>();
      ICHECK(attr_stmt);
      auto analyzer = arith::Analyzer();
      ICHECK(attr_stmt->attr_key == tir::attr::thread_extent &&
             StartsWith(Downcast<IterVar>(attr_stmt->node)->thread_tag, "blockIdx.") &&
             analyzer.CanProveEqual(attr_stmt->value, collector_.get_block_sum_extent()));
      auto body = attr_stmt->body;
      if (merged_attrs_body.empty()) {
        // keep one of the AttrStmt
        first_attr_stmt = Downcast<AttrStmt>(stmt);
      }
      merged_attrs_body.push_back(body);
    }
    new_seq.push_back(
        AttrStmt(first_attr_stmt->node, first_attr_stmt->attr_key, first_attr_stmt->value,
                 SeqStmt(merged_attrs_body, first_attr_stmt->span), first_attr_stmt->span));
    // after merge end
    // risk: if indices are not continuous, order may be wrong
    std::unordered_set<size_t> merge_candidate_set(merge_candidate_indices.begin(),
                                                   merge_candidate_indices.end());
    for (size_t i = 0; i < op->seq.size(); ++i) {
      if (merge_candidate_set.count(i)) {
        continue;
      }
      new_seq.push_back(this->VisitStmt(op->seq[i]));
    }

    return SeqStmt::Flatten(new_seq);
  }

  PrimExpr VisitExpr_(const VarNode* op) override {
    if (!in_merge_scope_) {
      return StmtExprMutator::VisitExpr_(op);
    }
    Var var = GetRef<Var>(op);
    if (StartsWith(var->name_hint, "blockIdx.") && block_idx_var_.defined()) {
      return block_idx_var_->var - current_extent_;
    } else if (StartsWith(var->name_hint, "threadIdx.") && thread_idx_var_.defined()) {
      return thread_idx_var_->var;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  int current_extent_{0};
  IterVar block_idx_var_{nullptr};
  IterVar thread_idx_var_{nullptr};
  const ThreadInfoCollector collector_;
  bool in_merge_scope_{false};
};

class SyncCounter : public StmtVisitor {
 public:
  int64_t sync_times_{0};

 private:
  // Count the number of storage sync operations.
  void VisitStmt_(const ForNode* op) override {
    arith::Analyzer analyzer;
    PrimExpr extent = analyzer.Simplify(op->extent);
    if (const IntImmNode* imm = extent.as<IntImmNode>()) {
      loop_multiplier_ *= imm->value;
      StmtVisitor::VisitStmt_(op);
      loop_multiplier_ /= imm->value;
    } else {
      // I don't know whether the case is possible
      StmtVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const EvaluateNode* op) override {
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        sync_times_ += loop_multiplier_;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  int64_t loop_multiplier_ = 1;
};

class InjectDivergentThreadSync : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == merge_candidate_key) {
      in_merge_scope_ = true;
      Stmt body = this->VisitStmt(op->body);
      in_merge_scope_ = false;
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    if (in_merge_scope_ && op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      // Only for 1-dim blockIdx
      if (iv->thread_tag == "blockIdx.x") {
        // Collect the number of sync operations in the body.
        SyncCounter sync_counter;
        sync_counter(op->body);
        sync_counter_ = std::move(sync_counter);
      } else if (iv->thread_tag == "threadIdx.x") {
        Stmt body = this->VisitStmt(op->body);
        Stmt guarded_body =
            IfThenElse(GE(iv, op->value),
                       For(Var(), 0, Integer(sync_counter_.sync_times_), ForKind::kSerial,
                           Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                                         {StringImm("shared")}))),
                       body);
        return AttrStmt(op->node, op->attr_key, op->value, guarded_body, op->span);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  SyncCounter sync_counter_;
  bool in_merge_scope_{false};
};

class CandidateMarker : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const ForNode* op) override {
    if (has_marked_) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (op->kind == ForKind::kThreadBinding) {
      auto iv = Downcast<IterVar>(op->thread_binding);
      if (iv->thread_tag == "blockIdx.x") {
        has_marked_ = true;
        auto body = StmtExprMutator::VisitStmt_(op);
        return AttrStmt(PrimExpr(0), merge_candidate_key, Integer(0), body, op->span);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  bool has_marked_{false};
};

namespace transform {

Pass MergeLaunchThread() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    // inject thread sync for divergent threads
    n->body = InjectDivergentThreadSync()(std::move(n->body));

    // merge blockIdx into a single one
    // and threadIdx into a single one
    ThreadInfoCollector collector;
    collector(n->body);
    n->body = LaunchThreadMerger(collector)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.MergeLaunchThread", {});
}

TVM_FFI_REGISTER_GLOBAL("tir.transform.MergeLaunchThread").set_body_typed(MergeLaunchThread);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.merge_launch_thread", Bool);

Pass MarkMergeCandidate() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = CandidateMarker()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.MarkMergeCandidate", {});
}

TVM_FFI_REGISTER_GLOBAL("tir.transform.MarkMergeCandidate").set_body_typed(MarkMergeCandidate);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
