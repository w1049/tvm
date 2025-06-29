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

class ThreadInfoCollector : public StmtVisitor {
 public:
  bool has_multiple_block_idx{false};
  PrimExpr sum_block_extent{nullptr};
  bool has_different_thread_idx{false};
  PrimExpr max_thread_extent{nullptr};

 private:
  // Visit the AttrStmt nodes to collect thread extent information.
  void VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (StartsWith(iv->thread_tag, "blockIdx.")) {
        if (!sum_block_extent.defined()) {
          sum_block_extent = op->value;
        } else {
          has_multiple_block_idx = true;
          sum_block_extent = sum_block_extent + op->value;
        }
      } else if (StartsWith(iv->thread_tag, "threadIdx.")) {
        if (!max_thread_extent.defined()) {
          max_thread_extent = op->value;
        } else {
          if (!max_thread_extent.same_as(op->value)) {
            has_different_thread_idx = true;
            max_thread_extent = tvm::max(max_thread_extent, op->value);
          }
        }
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
};

class MergeLaunchThread : public StmtExprMutator {
 public:
  explicit MergeLaunchThread(const ThreadInfoCollector& collector) : collector_(collector) {}

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) override {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (StartsWith(iv->thread_tag, "blockIdx.")) {
        if (!collector_.has_multiple_block_idx) {
          // If there is only one blockIdx, we can keep it as is.
          return StmtExprMutator::VisitStmt_(op);
        }
        // If there are multiple blockIdx, we need to merge them.
        // In VisitExpr, subtract the current extent from the blockIdx variable.

        // All blockIdx variables should be merged into a single one.
        if (!block_idx_var_.defined()) {
          // Do all range start at 0?
          block_idx_var_ = IterVar(Range::FromMinExtent(0, collector_.sum_block_extent), iv->var,
                                   iv->iter_type, iv->thread_tag, iv->span);
        }

        auto body = this->VisitStmt(op->body);
        auto start_extent = current_extent_;
        current_extent_ += op->value;
        return AttrStmt(
            block_idx_var_, op->attr_key, collector_.sum_block_extent,
            IfThenElse(And(GE(block_idx_var_, start_extent), LT(block_idx_var_, current_extent_)),
                       body),
            op->span);
      } else if (StartsWith(iv->thread_tag, "threadIdx.")) {
        if (!collector_.has_different_thread_idx) {
          // If all threadIdx variables have the same extent, we can keep it as is.
          return StmtExprMutator::VisitStmt_(op);
        }
        // If then else has been placed by InjectDivergentThreadSync,
        // we need to adjust the body to use the max_thread_extent.
        auto body = this->VisitStmt(op->body);
        // Create a new IterVar with the max_thread_extent.
        auto new_iv = IterVar(Range::FromMinExtent(0, collector_.max_thread_extent), iv->var,
                              iv->iter_type, iv->thread_tag, iv->span);
        return AttrStmt(new_iv, op->attr_key, collector_.max_thread_extent, body, op->span);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    // in my use case, top level is SeqStmt { AttrStmt, AttrStmt }
    // check if all attr statements are adjacent to each other
    int first_attr_idx = -1;
    int last_attr_idx = -1;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      if (const auto* attr_stmt = op->seq[i].as<AttrStmtNode>()) {
        if (attr_stmt->attr_key == tir::attr::thread_extent &&
            StartsWith(Downcast<IterVar>(attr_stmt->node)->thread_tag, "blockIdx.")) {
          if (first_attr_idx == -1) {
            first_attr_idx = i;
          }
          last_attr_idx = i;
        }
      }
    }

    bool all_adjacent = false;
    if (first_attr_idx != -1 && last_attr_idx != -1) {
      all_adjacent = true;
      for (int i = first_attr_idx; i <= last_attr_idx; ++i) {
        if (const auto* attr_stmt = op->seq[i].as<AttrStmtNode>()) {
          if (attr_stmt->attr_key != tir::attr::thread_extent ||
              !StartsWith(Downcast<IterVar>(attr_stmt->node)->thread_tag, "blockIdx.")) {
            all_adjacent = false;
            break;
          }
        } else {
          all_adjacent = false;
          break;
        }
      }
    }

    if (!all_adjacent) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // If all attr statements are adjacent, we can merge them.
    Array<Stmt> new_seq;
    for (int i = 0; i < first_attr_idx; ++i) {
      new_seq.push_back(this->VisitStmt(op->seq[i]));
    }

    Array<Stmt> merged_attrs_body;
    AttrStmt first_attr_stmt;
    for (int i = first_attr_idx; i <= last_attr_idx; ++i) {
      Stmt stmt = this->VisitStmt(op->seq[i]);
      auto attr_stmt = stmt.as<AttrStmtNode>();
      ICHECK(attr_stmt);
      ICHECK(attr_stmt->attr_key == tir::attr::thread_extent &&
             StartsWith(Downcast<IterVar>(attr_stmt->node)->thread_tag, "blockIdx.") &&
             attr_stmt->value.same_as(collector_.sum_block_extent));
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

    for (size_t i = last_attr_idx + 1; i < op->seq.size(); ++i) {
      new_seq.push_back(this->VisitStmt(op->seq[i]));
    }

    return SeqStmt::Flatten(new_seq);
  }

  PrimExpr VisitExpr_(const VarNode* op) override {
    Var var = GetRef<Var>(op);
    if (StartsWith(var->name_hint, "blockIdx.") && block_idx_var_.defined()) {
      return block_idx_var_->var - current_extent_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr current_extent_{0};
  IterVar block_idx_var_{nullptr};
  const ThreadInfoCollector collector_;
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
    if (op->attr_key == tir::attr::thread_extent) {
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
};

namespace transform {

Pass MergeLaunchThreadPass() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    // inject thread sync for divergent threads
    n->body = InjectDivergentThreadSync()(std::move(n->body));

    // merge blockIdx into a single one
    // and threadIdx into a single one
    ThreadInfoCollector collector;
    collector(n->body);
    n->body = MergeLaunchThread(collector)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.MergeLaunchThreadPass", {});
}

TVM_FFI_REGISTER_GLOBAL("tir.transform.MergeLaunchThreadPass")
    .set_body_typed(MergeLaunchThreadPass);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.merge_launch_thread", Bool);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
