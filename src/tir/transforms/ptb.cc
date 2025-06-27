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

#include <tvm/ffi/function.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class SyncReplacer : public StmtExprMutator {
 private:
  PrimExpr VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(builtin::tvm_storage_sync())) {
      auto new_arg = StringImm("block." + op->args[0].as<StringImmNode>()->value);
      return Call(op->dtype, op->op, {new_arg}, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};

namespace transform {

Pass ThreadBlockSync() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = SyncReplacer()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ThreadBlockSync", {});
}

TVM_FFI_REGISTER_GLOBAL("tir.transform.ThreadBlockSync").set_body_typed(ThreadBlockSync);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.use_block_sync", Bool);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
