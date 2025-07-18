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
#include "../module_equality.h"
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/******** Workload ********/

Workload::Workload(IRModule mod) {
  ObjectPtr<WorkloadNode> n = ffi::make_object<WorkloadNode>();
  n->mod = mod;
  n->shash = ModuleEquality::Create("structural")->Hash(mod);
  data_ = std::move(n);
}

Workload::Workload(IRModule mod, Workload::THashCode shash) {
  ObjectPtr<WorkloadNode> n = ffi::make_object<WorkloadNode>();
  n->mod = mod;
  n->shash = shash;
  data_ = std::move(n);
}

ObjectRef WorkloadNode::AsJSON() const {
  // Convert `this->mod` to JSON
  std::string json_mod = tvm::SaveJSON(this->mod);
  // Dump the JSON string to base64
  std::string b64_mod = Base64Encode(json_mod);
  // Output
  return Array<ObjectRef>{SHash2Str(this->shash), String(b64_mod)};
}

Workload Workload::FromJSON(const ObjectRef& json_obj) {
  IRModule mod{nullptr};
  THashCode shash = 0;
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    CHECK(json_array && json_array->size() == 2);
    // Load json[0] => shash
    String str_shash = json_array->at(0).cast<String>();
    // Load json[1] => mod
    {
      String b64_mod = json_array->at(1).cast<String>();
      std::string json_mod = Base64Decode(b64_mod);
      mod = LoadJSON(json_mod).cast<IRModule>();
      std::stringstream(str_shash) >> shash;
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return Workload(mod, shash);
}

/******** TuningRecord ********/

TuningRecord::TuningRecord(tir::Trace trace, Workload workload, Optional<Array<FloatImm>> run_secs,
                           Optional<Target> target, Optional<Array<ArgInfo>> args_info, Optional<Array<Integer>> extra_info) {
  ObjectPtr<TuningRecordNode> n = make_object<TuningRecordNode>();
  n->trace = trace;
  n->workload = workload;
  n->run_secs = run_secs;
  n->target = target;
  n->args_info = args_info;
  n->extra_info = extra_info;
  this->data_ = n;
}

bool WorkloadEqual::operator()(const Workload& a, const Workload& b) const {
  return a->shash == b->shash && mod_eq_.Equal(a->mod, b->mod);
}

MeasureCandidate TuningRecordNode::AsMeasureCandidate() const {
  tir::Schedule sch =
      tir::Schedule::Traced(workload->mod, -1, 0, tir::ScheduleErrorRenderLevel::kDetail);
  trace->ApplyToSchedule(sch, false, nullptr);
  return MeasureCandidate(sch, ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true));
}

ObjectRef TuningRecordNode::AsJSON() const {
  Optional<Array<ObjectRef>> json_args_info;
  Optional<ObjectRef> json_target;
  if (args_info.defined()) {
    Array<ObjectRef> info;
    info.reserve(args_info.value().size());
    for (const ArgInfo& arg_info : args_info.value()) {
      info.push_back(arg_info->AsJSON());
    }
    json_args_info = info;
  }
  if (target.defined()) {
    json_target = target.value()->Export();
  }
  return Array<ObjectRef>{trace->AsJSON(false),  //
                          run_secs,              //
                          json_target,           //
                          json_args_info,        //
                          extra_info};
}

bool TuningRecordNode::IsValid() const {
  if (!GetNumValidInstructions(trace->insts, /*remove_postproc*/ true)) {
    return false;
  }
  if (run_secs.defined()) {
    for (const auto& run_sec : run_secs.value()) {
      // kMaxMeanTime(1e10) is used as a stub for undefined measurement times.
      if (run_sec.defined() && run_sec->value != SortTuningRecordByMeanRunSecs::kMaxMeanTime) {
        return true;
      }
    }
  }
  return false;
}

TuningRecord TuningRecord::FromJSON(const ObjectRef& json_obj, const Workload& workload) {
  tir::Trace trace{nullptr};
  Optional<Array<FloatImm>> run_secs;
  Optional<Target> target;
  Optional<Array<ArgInfo>> args_info;
  Optional<Array<Integer>> extra_info;
  try {
    const ffi::ArrayObj* json_array = json_obj.as<ffi::ArrayObj>();
    CHECK(json_array && (json_array->size() == 4 || json_array->size() == 5));
    // Load json[1] => run_secs
    if (json_array->at(1) != nullptr) {
      run_secs = AsFloatArray(json_array->at(1).cast<ObjectRef>());
    }
    // Load json[2] => target
    if (json_array->at(2) != nullptr) {
      target = Target(json_array->at(2).cast<Map<String, ffi::Any>>());
    }
    // Load json[3] => args_info
    if (json_array->at(3) != nullptr) {
      const ffi::ArrayObj* json_args_info = json_array->at(3).cast<const ffi::ArrayObj*>();
      Array<ArgInfo> info;
      info.reserve(json_args_info->size());
      for (Any json_arg_info : *json_args_info) {
        info.push_back(ArgInfo::FromJSON(json_arg_info.cast<ObjectRef>()));
      }
      args_info = info;
    }
    // Load json[0] => trace
    {
      auto json_trace = json_array->at(0).cast<ObjectRef>();
      tir::Schedule sch =
          tir::Schedule::Traced(workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                                /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
      tir::Trace::ApplyJSONToSchedule(json_trace, sch);
      trace = sch->trace().value();
    }
    if (json_array->size() == 5) {
      // Load json[4] => extra_info
      extra_info = AsIntArray(json_array->at(4).cast<ObjectRef>());
    }
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TuningRecord(trace, workload, run_secs, target, args_info, extra_info);
}

/******** Database ********/
DatabaseNode::DatabaseNode(String mod_eq_name) { mod_eq_ = ModuleEquality::Create(mod_eq_name); }
DatabaseNode::~DatabaseNode() = default;

Optional<TuningRecord> DatabaseNode::QueryTuningRecord(const IRModule& mod, const Target& target,
                                                       const String& workload_name) {
  if (!this->HasWorkload(mod)) {
    return std::nullopt;
  }
  Array<TuningRecord> records = this->GetTopK(this->CommitWorkload(mod), 1);
  if (records.empty()) {
    return std::nullopt;
  }
  ICHECK_EQ(records.size(), 1);
  return records[0];
}

Optional<tir::Schedule> DatabaseNode::QuerySchedule(const IRModule& mod, const Target& target,
                                                    const String& workload_name) {
  if (Optional<TuningRecord> opt_record = this->QueryTuningRecord(mod, target, workload_name)) {
    TuningRecord record = opt_record.value();
    tir::Schedule sch =
        tir::Schedule::Traced(record->workload->mod, /*seed=*/-1, /*debug_mask=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    record->trace->ApplyToSchedule(sch, false);
    return sch;
  } else {
    return std::nullopt;
  }
}

Optional<IRModule> DatabaseNode::QueryIRModule(const IRModule& mod, const Target& target,
                                               const String& workload_name) {
  if (Optional<tir::Schedule> opt_sch = this->QuerySchedule(mod, target, workload_name)) {
    return opt_sch.value()->mod();
  } else {
    return std::nullopt;
  }
}

void DatabaseNode::DumpPruned(Database destination) {
  std::unordered_map<Workload, TuningRecord, ObjectPtrHash, ObjectPtrEqual> workload2record;
  for (const TuningRecord& record : this->GetAllTuningRecords()) {
    if (record->IsValid()) {
      auto it = workload2record.find(record->workload);
      if (it == workload2record.end()) {
        workload2record.insert({record->workload, record});
      } else if (SortTuningRecordByMeanRunSecs()(record, it->second)) {
        it->second = record;
      }
    }
  }
  for (auto& kv : workload2record) {
    Workload workload = kv.first;
    TuningRecord record = kv.second;
    workload = destination->CommitWorkload(workload->mod);
    destination->CommitTuningRecord(TuningRecord(/*trace=*/record->trace, /*workload=*/workload,
                                                 /*run_secs=*/record->run_secs,
                                                 /*target=*/record->target,
                                                 /*args_info=*/record->args_info,
                                                /*extra_info=*/record->extra_info));
  }
}

std::vector<Database>* ThreadLocalDatabases() {
  static thread_local std::vector<Database> tls;
  return &tls;
}

void Database::EnterWithScope() { ThreadLocalDatabases()->push_back(*this); }

void Database::ExitWithScope() { ThreadLocalDatabases()->pop_back(); }

Optional<Database> Database::Current() {
  std::vector<Database>* tls = ThreadLocalDatabases();
  if (tls->empty()) {
    return std::nullopt;
  } else {
    return tls->back();
  }
}

/******** PyDatabase ********/
PyDatabaseNode::PyDatabaseNode(String mod_eq_name) : DatabaseNode(mod_eq_name) {}

Database Database::PyDatabase(PyDatabaseNode::FHasWorkload f_has_workload,
                              PyDatabaseNode::FCommitWorkload f_commit_workload,
                              PyDatabaseNode::FCommitTuningRecord f_commit_tuning_record,
                              PyDatabaseNode::FGetTopK f_get_top_k,
                              PyDatabaseNode::FGetAllTuningRecords f_get_all_tuning_records,
                              PyDatabaseNode::FQueryTuningRecord f_query_tuning_record,
                              PyDatabaseNode::FQuerySchedule f_query_schedule,
                              PyDatabaseNode::FQueryIRModule f_query_ir_module,
                              PyDatabaseNode::FSize f_size, String mod_eq_name) {
  ObjectPtr<PyDatabaseNode> n = make_object<PyDatabaseNode>(mod_eq_name);
  n->f_has_workload = f_has_workload;
  n->f_commit_workload = f_commit_workload;
  n->f_commit_tuning_record = f_commit_tuning_record;
  n->f_get_top_k = f_get_top_k;
  n->f_get_all_tuning_records = f_get_all_tuning_records;
  n->f_query_tuning_record = f_query_tuning_record;
  n->f_query_schedule = f_query_schedule;
  n->f_query_ir_module = f_query_ir_module;
  n->f_size = f_size;
  return Database(n);
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK({
  WorkloadNode::RegisterReflection();
  TuningRecordNode::RegisterReflection();
  PyDatabaseNode::RegisterReflection();
});

TVM_REGISTER_NODE_TYPE(WorkloadNode);
TVM_REGISTER_NODE_TYPE(TuningRecordNode);
TVM_REGISTER_OBJECT_TYPE(DatabaseNode);
TVM_REGISTER_NODE_TYPE(PyDatabaseNode);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.Workload").set_body_typed([](IRModule mod) {
  return Workload(mod);
});
TVM_FFI_REGISTER_GLOBAL("meta_schedule.WorkloadAsJSON").set_body_method(&WorkloadNode::AsJSON);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.WorkloadFromJSON").set_body_typed(&Workload::FromJSON);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.TuningRecord")
    .set_body_typed([](tir::Trace trace, Workload workload, Optional<Array<FloatImm>> run_secs,
                       Optional<Target> target, Optional<Array<ArgInfo>> args_info,
                       Optional<Array<Integer>> extra_info) {
      return TuningRecord(trace, workload, run_secs, target, args_info, extra_info);
    });
TVM_FFI_REGISTER_GLOBAL("meta_schedule.TuningRecordAsMeasureCandidate")
    .set_body_method(&TuningRecordNode::AsMeasureCandidate);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.TuningRecordAsJSON")
    .set_body_method(&TuningRecordNode::AsJSON);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.TuningRecordFromJSON")
    .set_body_typed(TuningRecord::FromJSON);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseEnterWithScope")
    .set_body_method(&Database::EnterWithScope);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseExitWithScope")
    .set_body_method(&Database::ExitWithScope);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseCurrent").set_body_typed(Database::Current);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseHasWorkload")
    .set_body_method(&DatabaseNode::HasWorkload);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseCommitWorkload")
    .set_body_method(&DatabaseNode::CommitWorkload);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseCommitTuningRecord")
    .set_body_method(&DatabaseNode::CommitTuningRecord);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseGetTopK").set_body_method(&DatabaseNode::GetTopK);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseGetAllTuningRecords")
    .set_body_method(&DatabaseNode::GetAllTuningRecords);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseSize").set_body_method(&DatabaseNode::Size);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseQueryTuningRecord")
    .set_body_method(&DatabaseNode::QueryTuningRecord);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseQuerySchedule")
    .set_body_method(&DatabaseNode::QuerySchedule);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseQueryIRModule")
    .set_body_method(&DatabaseNode::QueryIRModule);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabaseDumpPruned")
    .set_body_method(&DatabaseNode::DumpPruned);
TVM_FFI_REGISTER_GLOBAL("meta_schedule.DatabasePyDatabase").set_body_typed(Database::PyDatabase);

}  // namespace meta_schedule
}  // namespace tvm
