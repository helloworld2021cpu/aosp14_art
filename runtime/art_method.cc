/*
 * Copyright (C) 2011 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "art_method.h"

#include <algorithm>
#include <cstddef>

#include "android-base/stringprintf.h"

#include "arch/context.h"
#include "art_method-inl.h"
#include "base/enums.h"
#include "base/stl_util.h"
#include "class_linker-inl.h"
#include "class_root-inl.h"
#include "debugger.h"
#include "dex/class_accessor-inl.h"
#include "dex/descriptors_names.h"
#include "dex/dex_file-inl.h"
#include "dex/dex_file_exception_helpers.h"
#include "dex/dex_instruction.h"
#include "dex/signature-inl.h"
#include "entrypoints/runtime_asm_entrypoints.h"
#include "gc/accounting/card_table-inl.h"
#include "hidden_api.h"
#include "interpreter/interpreter.h"
#include "jit/jit.h"
#include "jit/jit_code_cache.h"
#include "jit/profiling_info.h"
#include "jni/jni_internal.h"
#include "mirror/class-inl.h"
#include "mirror/class_ext-inl.h"
#include "mirror/executable.h"
#include "mirror/object-inl.h"
#include "mirror/object_array-inl.h"
#include "mirror/string.h"
#include "oat_file-inl.h"
#include "quicken_info.h"
#include "runtime_callbacks.h"
#include "scoped_thread_state_change-inl.h"
#include "vdex_file.h"

#include "base/arena_allocator.h"
#include "base/malloc_arena_pool.h"
#include "compiler/jni/quick/calling_convention.h"
#include "arch/instruction_set.h"
#include <sys/syscall.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "nth_caller_visitor.h"

namespace art {

using android::base::StringPrintf;

extern "C" void art_quick_invoke_stub(ArtMethod*, uint32_t*, uint32_t, Thread*, JValue*,
                                      const char*);
extern "C" void art_quick_invoke_static_stub(ArtMethod*, uint32_t*, uint32_t, Thread*, JValue*,
                                             const char*);
extern "C" void MyWrite(unsigned char *pdata, int nlen, const char *pflag, int tid);
extern "C" pid_t gettid(void);
extern "C" void DumpHex(const void *vdata, size_t size, int tid);
extern "C" void TestArg(ArtMethod* m, uint32_t* args);
extern "C" void TestJniArg(ArtMethod *method, Thread *self, void *sp);

bool bTrace = false;

// Enforce that we have the right index for runtime methods.
static_assert(ArtMethod::kRuntimeMethodDexMethodIndex == dex::kDexNoIndex,
              "Wrong runtime-method dex method index");

ArtMethod* ArtMethod::GetCanonicalMethod(PointerSize pointer_size) {
  if (LIKELY(!IsCopied())) {
    return this;
  } else {
    ObjPtr<mirror::Class> declaring_class = GetDeclaringClass();
    DCHECK(declaring_class->IsInterface());
    ArtMethod* ret = declaring_class->FindInterfaceMethod(GetDexCache(),
                                                          GetDexMethodIndex(),
                                                          pointer_size);
    DCHECK(ret != nullptr);
    return ret;
  }
}

ArtMethod* ArtMethod::GetNonObsoleteMethod() {
  if (LIKELY(!IsObsolete())) {
    return this;
  }
  DCHECK_EQ(kRuntimePointerSize, Runtime::Current()->GetClassLinker()->GetImagePointerSize());
  if (IsDirect()) {
    return &GetDeclaringClass()->GetDirectMethodsSlice(kRuntimePointerSize)[GetMethodIndex()];
  } else {
    return GetDeclaringClass()->GetVTableEntry(GetMethodIndex(), kRuntimePointerSize);
  }
}

ArtMethod* ArtMethod::GetSingleImplementation(PointerSize pointer_size) {
  if (IsInvokable()) {
    // An invokable method single implementation is itself.
    return this;
  }
  DCHECK(!IsDefaultConflicting());
  ArtMethod* m = reinterpret_cast<ArtMethod*>(GetDataPtrSize(pointer_size));
  CHECK(m == nullptr || !m->IsDefaultConflicting());
  return m;
}

ArtMethod* ArtMethod::FromReflectedMethod(const ScopedObjectAccessAlreadyRunnable& soa,
                                          jobject jlr_method) {
  ObjPtr<mirror::Executable> executable = soa.Decode<mirror::Executable>(jlr_method);
  DCHECK(executable != nullptr);
  return executable->GetArtMethod();
}

template <ReadBarrierOption kReadBarrierOption>
ObjPtr<mirror::DexCache> ArtMethod::GetObsoleteDexCache() {
  // Note: The class redefinition happens with GC disabled, so at the point where we
  // create obsolete methods, the `ClassExt` and its obsolete methods and dex caches
  // members are reachable without a read barrier. If we start a GC later, and we
  // look at these objects without read barriers (`kWithoutReadBarrier`), the method
  // pointers shall be the same in from-space array as in to-space array (if these
  // arrays are different) and the dex cache array entry can point to from-space or
  // to-space `DexCache` but either is a valid result for `kWithoutReadBarrier`.
  ScopedAssertNoThreadSuspension ants(__FUNCTION__);
  std::optional<ScopedDebugDisallowReadBarriers> sddrb(std::nullopt);
  if (kIsDebugBuild && kReadBarrierOption == kWithoutReadBarrier) {
    sddrb.emplace(Thread::Current());
  }
  PointerSize pointer_size = kRuntimePointerSize;
  DCHECK(!Runtime::Current()->IsAotCompiler()) << PrettyMethod();
  DCHECK(IsObsolete());
  ObjPtr<mirror::Class> declaring_class = GetDeclaringClass<kReadBarrierOption>();
  ObjPtr<mirror::ClassExt> ext =
      declaring_class->GetExtData<kDefaultVerifyFlags, kReadBarrierOption>();
  ObjPtr<mirror::PointerArray> obsolete_methods(
      ext.IsNull() ? nullptr : ext->GetObsoleteMethods<kDefaultVerifyFlags, kReadBarrierOption>());
  int32_t len = 0;
  ObjPtr<mirror::ObjectArray<mirror::DexCache>> obsolete_dex_caches = nullptr;
  if (!obsolete_methods.IsNull()) {
    len = obsolete_methods->GetLength();
    obsolete_dex_caches = ext->GetObsoleteDexCaches<kDefaultVerifyFlags, kReadBarrierOption>();
    // FIXME: `ClassExt::SetObsoleteArrays()` is not atomic, so one of the arrays we see here
    // could be extended for a new class redefinition while the other may be shorter.
    // Furthermore, there is no synchronization to ensure that copied contents of an old
    // obsolete array are visible to a thread reading the new array.
    DCHECK_EQ(len, obsolete_dex_caches->GetLength())
        << " ext->GetObsoleteDexCaches()=" << obsolete_dex_caches;
  }
  // Using kRuntimePointerSize (instead of using the image's pointer size) is fine since images
  // should never have obsolete methods in them so they should always be the same.
  DCHECK_EQ(pointer_size, Runtime::Current()->GetClassLinker()->GetImagePointerSize());
  for (int32_t i = 0; i < len; i++) {
    if (this == obsolete_methods->GetElementPtrSize<ArtMethod*>(i, pointer_size)) {
      return obsolete_dex_caches->GetWithoutChecks<kDefaultVerifyFlags, kReadBarrierOption>(i);
    }
  }
  CHECK(declaring_class->IsObsoleteObject())
      << "This non-structurally obsolete method does not appear in the obsolete map of its class: "
      << declaring_class->PrettyClass() << " Searched " << len << " caches.";
  CHECK_EQ(this,
           std::clamp(this,
                      &(*declaring_class->GetMethods(pointer_size).begin()),
                      &(*declaring_class->GetMethods(pointer_size).end())))
      << "class is marked as structurally obsolete method but not found in normal obsolete-map "
      << "despite not being the original method pointer for " << GetDeclaringClass()->PrettyClass();
  return declaring_class->template GetDexCache<kDefaultVerifyFlags, kReadBarrierOption>();
}

template ObjPtr<mirror::DexCache> ArtMethod::GetObsoleteDexCache<kWithReadBarrier>();
template ObjPtr<mirror::DexCache> ArtMethod::GetObsoleteDexCache<kWithoutReadBarrier>();

uint16_t ArtMethod::FindObsoleteDexClassDefIndex() {
  DCHECK(!Runtime::Current()->IsAotCompiler()) << PrettyMethod();
  DCHECK(IsObsolete());
  const DexFile* dex_file = GetDexFile();
  const dex::TypeIndex declaring_class_type = dex_file->GetMethodId(GetDexMethodIndex()).class_idx_;
  const dex::ClassDef* class_def = dex_file->FindClassDef(declaring_class_type);
  CHECK(class_def != nullptr);
  return dex_file->GetIndexForClassDef(*class_def);
}

void ArtMethod::ThrowInvocationTimeError(ObjPtr<mirror::Object> receiver) {
  DCHECK(!IsInvokable());
  if (IsDefaultConflicting()) {
    ThrowIncompatibleClassChangeErrorForMethodConflict(this);
  } else if (GetDeclaringClass()->IsInterface() && receiver != nullptr) {
    // If this was an interface call, check whether there is a method in the
    // superclass chain that isn't public. In this situation, we should throw an
    // IllegalAccessError.
    DCHECK(IsAbstract());
    ObjPtr<mirror::Class> current = receiver->GetClass();
    while (current != nullptr) {
      for (ArtMethod& method : current->GetDeclaredMethodsSlice(kRuntimePointerSize)) {
        ArtMethod* np_method = method.GetInterfaceMethodIfProxy(kRuntimePointerSize);
        if (!np_method->IsStatic() &&
            np_method->GetNameView() == GetNameView() &&
            np_method->GetSignature() == GetSignature()) {
          if (!np_method->IsPublic()) {
            ThrowIllegalAccessErrorForImplementingMethod(receiver->GetClass(), np_method, this);
            return;
          } else if (np_method->IsAbstract()) {
            ThrowAbstractMethodError(this);
            return;
          }
        }
      }
      current = current->GetSuperClass();
    }
    ThrowAbstractMethodError(this);
  } else {
    DCHECK(IsAbstract());
    ThrowAbstractMethodError(this);
  }
}

InvokeType ArtMethod::GetInvokeType() {
  // TODO: kSuper?
  if (IsStatic()) {
    return kStatic;
  } else if (GetDeclaringClass()->IsInterface()) {
    return kInterface;
  } else if (IsDirect()) {
    return kDirect;
  } else if (IsSignaturePolymorphic()) {
    return kPolymorphic;
  } else {
    return kVirtual;
  }
}

size_t ArtMethod::NumArgRegisters(const char* shorty) {
  CHECK_NE(shorty[0], '\0');
  uint32_t num_registers = 0;
  for (const char* s = shorty + 1; *s != '\0'; ++s) {
    if (*s == 'D' || *s == 'J') {
      num_registers += 2;
    } else {
      num_registers += 1;
    }
  }
  return num_registers;
}

bool ArtMethod::HasSameNameAndSignature(ArtMethod* other) {
  ScopedAssertNoThreadSuspension ants("HasSameNameAndSignature");
  const DexFile* dex_file = GetDexFile();
  const dex::MethodId& mid = dex_file->GetMethodId(GetDexMethodIndex());
  if (GetDexCache() == other->GetDexCache()) {
    const dex::MethodId& mid2 = dex_file->GetMethodId(other->GetDexMethodIndex());
    return mid.name_idx_ == mid2.name_idx_ && mid.proto_idx_ == mid2.proto_idx_;
  }
  const DexFile* dex_file2 = other->GetDexFile();
  const dex::MethodId& mid2 = dex_file2->GetMethodId(other->GetDexMethodIndex());
  if (!DexFile::StringEquals(dex_file, mid.name_idx_, dex_file2, mid2.name_idx_)) {
    return false;  // Name mismatch.
  }
  return dex_file->GetMethodSignature(mid) == dex_file2->GetMethodSignature(mid2);
}

ArtMethod* ArtMethod::FindOverriddenMethod(PointerSize pointer_size) {
  if (IsStatic()) {
    return nullptr;
  }
  ObjPtr<mirror::Class> declaring_class = GetDeclaringClass();
  ObjPtr<mirror::Class> super_class = declaring_class->GetSuperClass();
  uint16_t method_index = GetMethodIndex();
  ArtMethod* result = nullptr;
  // Did this method override a super class method? If so load the result from the super class'
  // vtable
  if (super_class->HasVTable() && method_index < super_class->GetVTableLength()) {
    result = super_class->GetVTableEntry(method_index, pointer_size);
  } else {
    // Method didn't override superclass method so search interfaces
    if (IsProxyMethod()) {
      result = GetInterfaceMethodIfProxy(pointer_size);
      DCHECK(result != nullptr);
    } else {
      ObjPtr<mirror::IfTable> iftable = GetDeclaringClass()->GetIfTable();
      for (size_t i = 0; i < iftable->Count() && result == nullptr; i++) {
        ObjPtr<mirror::Class> interface = iftable->GetInterface(i);
        for (ArtMethod& interface_method : interface->GetVirtualMethods(pointer_size)) {
          if (HasSameNameAndSignature(interface_method.GetInterfaceMethodIfProxy(pointer_size))) {
            result = &interface_method;
            break;
          }
        }
      }
    }
  }
  DCHECK(result == nullptr ||
         GetInterfaceMethodIfProxy(pointer_size)->HasSameNameAndSignature(
             result->GetInterfaceMethodIfProxy(pointer_size)));
  return result;
}

uint32_t ArtMethod::FindDexMethodIndexInOtherDexFile(const DexFile& other_dexfile,
                                                     uint32_t name_and_signature_idx) {
  const DexFile* dexfile = GetDexFile();
  const uint32_t dex_method_idx = GetDexMethodIndex();
  const dex::MethodId& mid = dexfile->GetMethodId(dex_method_idx);
  const dex::MethodId& name_and_sig_mid = other_dexfile.GetMethodId(name_and_signature_idx);
  DCHECK_STREQ(dexfile->GetMethodName(mid), other_dexfile.GetMethodName(name_and_sig_mid));
  DCHECK_EQ(dexfile->GetMethodSignature(mid), other_dexfile.GetMethodSignature(name_and_sig_mid));
  if (dexfile == &other_dexfile) {
    return dex_method_idx;
  }
  const char* mid_declaring_class_descriptor = dexfile->StringByTypeIdx(mid.class_idx_);
  const dex::TypeId* other_type_id = other_dexfile.FindTypeId(mid_declaring_class_descriptor);
  if (other_type_id != nullptr) {
    const dex::MethodId* other_mid = other_dexfile.FindMethodId(
        *other_type_id, other_dexfile.GetStringId(name_and_sig_mid.name_idx_),
        other_dexfile.GetProtoId(name_and_sig_mid.proto_idx_));
    if (other_mid != nullptr) {
      return other_dexfile.GetIndexForMethodId(*other_mid);
    }
  }
  return dex::kDexNoIndex;
}

uint32_t ArtMethod::FindCatchBlock(Handle<mirror::Class> exception_type,
                                   uint32_t dex_pc, bool* has_no_move_exception) {
  // Set aside the exception while we resolve its type.
  Thread* self = Thread::Current();
  StackHandleScope<1> hs(self);
  Handle<mirror::Throwable> exception(hs.NewHandle(self->GetException()));
  self->ClearException();
  // Default to handler not found.
  uint32_t found_dex_pc = dex::kDexNoIndex;
  // Iterate over the catch handlers associated with dex_pc.
  CodeItemDataAccessor accessor(DexInstructionData());
  for (CatchHandlerIterator it(accessor, dex_pc); it.HasNext(); it.Next()) {
    dex::TypeIndex iter_type_idx = it.GetHandlerTypeIndex();
    // Catch all case
    if (!iter_type_idx.IsValid()) {
      found_dex_pc = it.GetHandlerAddress();
      break;
    }
    // Does this catch exception type apply?
    ObjPtr<mirror::Class> iter_exception_type = ResolveClassFromTypeIndex(iter_type_idx);
    if (UNLIKELY(iter_exception_type == nullptr)) {
      // Now have a NoClassDefFoundError as exception. Ignore in case the exception class was
      // removed by a pro-guard like tool.
      // Note: this is not RI behavior. RI would have failed when loading the class.
      self->ClearException();
      // Delete any long jump context as this routine is called during a stack walk which will
      // release its in use context at the end.
      delete self->GetLongJumpContext();
      LOG(WARNING) << "Unresolved exception class when finding catch block: "
        << DescriptorToDot(GetTypeDescriptorFromTypeIdx(iter_type_idx));
    } else if (iter_exception_type->IsAssignableFrom(exception_type.Get())) {
      found_dex_pc = it.GetHandlerAddress();
      break;
    }
  }
  if (found_dex_pc != dex::kDexNoIndex) {
    const Instruction& first_catch_instr = accessor.InstructionAt(found_dex_pc);
    *has_no_move_exception = (first_catch_instr.Opcode() != Instruction::MOVE_EXCEPTION);
  }
  // Put the exception back.
  if (exception != nullptr) {
    self->SetException(exception.Get());
  }
  return found_dex_pc;
}

NO_STACK_PROTECTOR
void ArtMethod::Invoke(Thread* self, uint32_t* args, uint32_t args_size, JValue* result,
                       const char* shorty) {
  if (UNLIKELY(__builtin_frame_address(0) < self->GetStackEnd())) {
    ThrowStackOverflowError(self);
    return;
  }

  /*
  std::string output;
  output = this->PrettyMethod(true);
  MyWrite((unsigned char*)output.c_str(), output.size(), "[T+]:", gettid());
  */


  if (kIsDebugBuild) {
    self->AssertThreadSuspensionIsAllowable();
    CHECK_EQ(ThreadState::kRunnable, self->GetState());
    CHECK_STREQ(GetInterfaceMethodIfProxy(kRuntimePointerSize)->GetShorty(), shorty);
  }

  // Push a transition back into managed code onto the linked list in thread.
  ManagedStack fragment;
  self->PushManagedStackFragment(&fragment);

  Runtime* runtime = Runtime::Current();
  // Call the invoke stub, passing everything as arguments.
  // If the runtime is not yet started or it is required by the debugger, then perform the
  // Invocation by the interpreter, explicitly forcing interpretation over JIT to prevent
  // cycling around the various JIT/Interpreter methods that handle method invocation.
  if (UNLIKELY(!runtime->IsStarted() ||
               (self->IsForceInterpreter() && !IsNative() && !IsProxyMethod() && IsInvokable()))) {
    if (IsStatic()) {
      art::interpreter::EnterInterpreterFromInvoke(
          self, this, nullptr, args, result, /*stay_in_interpreter=*/ true);
    } else {
      mirror::Object* receiver =
          reinterpret_cast<StackReference<mirror::Object>*>(&args[0])->AsMirrorPtr();
      art::interpreter::EnterInterpreterFromInvoke(
          self, this, receiver, args + 1, result, /*stay_in_interpreter=*/ true);
    }
  } else {
    DCHECK_EQ(runtime->GetClassLinker()->GetImagePointerSize(), kRuntimePointerSize);

    constexpr bool kLogInvocationStartAndReturn = false;
    bool have_quick_code = GetEntryPointFromQuickCompiledCode() != nullptr;
    if (LIKELY(have_quick_code)) {
      if (kLogInvocationStartAndReturn) {
        LOG(INFO) << StringPrintf(
            "Invoking '%s' quick code=%p static=%d", PrettyMethod().c_str(),
            GetEntryPointFromQuickCompiledCode(), static_cast<int>(IsStatic() ? 1 : 0));
      }

      // Ensure that we won't be accidentally calling quick compiled code when -Xint.
      if (kIsDebugBuild && runtime->GetInstrumentation()->IsForcedInterpretOnly()) {
        CHECK(!runtime->UseJitCompilation());
        const void* oat_quick_code =
            (IsNative() || !IsInvokable() || IsProxyMethod() || IsObsolete())
            ? nullptr
            : GetOatMethodQuickCode(runtime->GetClassLinker()->GetImagePointerSize());
        CHECK(oat_quick_code == nullptr || oat_quick_code != GetEntryPointFromQuickCompiledCode())
            << "Don't call compiled code when -Xint " << PrettyMethod();
      }


      if (!IsStatic()) {
        (*art_quick_invoke_stub)(this, args, args_size, self, result, shorty);
      } else {
        (*art_quick_invoke_static_stub)(this, args, args_size, self, result, shorty);
      }
      if (UNLIKELY(self->GetException() == Thread::GetDeoptimizationException())) {
        // Unusual case where we were running generated code and an
        // exception was thrown to force the activations to be removed from the
        // stack. Continue execution in the interpreter.
        self->DeoptimizeWithDeoptimizationException(result);
      }
      if (kLogInvocationStartAndReturn) {
        LOG(INFO) << StringPrintf("Returned '%s' quick code=%p", PrettyMethod().c_str(),
                                  GetEntryPointFromQuickCompiledCode());
      }
    } else {
      LOG(INFO) << "Not invoking '" << PrettyMethod() << "' code=null";
      if (result != nullptr) {
        result->SetJ(0);
      }
    }
  }

  // Pop transition.
  self->PopManagedStackFragment(fragment);
}

bool ArtMethod::IsSignaturePolymorphic() {
  // Methods with a polymorphic signature have constraints that they
  // are native and varargs and belong to either MethodHandle or VarHandle.
  if (!IsNative() || !IsVarargs()) {
    return false;
  }
  ObjPtr<mirror::ObjectArray<mirror::Class>> class_roots =
      Runtime::Current()->GetClassLinker()->GetClassRoots();
  ObjPtr<mirror::Class> cls = GetDeclaringClass();
  return (cls == GetClassRoot<mirror::MethodHandle>(class_roots) ||
          cls == GetClassRoot<mirror::VarHandle>(class_roots));
}

static uint32_t GetOatMethodIndexFromMethodIndex(const DexFile& dex_file,
                                                 uint16_t class_def_idx,
                                                 uint32_t method_idx) {
  ClassAccessor accessor(dex_file, class_def_idx);
  uint32_t class_def_method_index = 0u;
  for (const ClassAccessor::Method& method : accessor.GetMethods()) {
    if (method.GetIndex() == method_idx) {
      return class_def_method_index;
    }
    class_def_method_index++;
  }
  LOG(FATAL) << "Failed to find method index " << method_idx << " in " << dex_file.GetLocation();
  UNREACHABLE();
}

// We use the method's DexFile and declaring class name to find the OatMethod for an obsolete
// method.  This is extremely slow but we need it if we want to be able to have obsolete native
// methods since we need this to find the size of its stack frames.
//
// NB We could (potentially) do this differently and rely on the way the transformation is applied
// in order to use the entrypoint to find this information. However, for debugging reasons (most
// notably making sure that new invokes of obsolete methods fail) we choose to instead get the data
// directly from the dex file.
static const OatFile::OatMethod FindOatMethodFromDexFileFor(ArtMethod* method, bool* found)
    REQUIRES_SHARED(Locks::mutator_lock_) {
  DCHECK(method->IsObsolete() && method->IsNative());
  const DexFile* dex_file = method->GetDexFile();

  // recreate the class_def_index from the descriptor.
  std::string descriptor_storage;
  const dex::TypeId* declaring_class_type_id =
      dex_file->FindTypeId(method->GetDeclaringClass()->GetDescriptor(&descriptor_storage));
  CHECK(declaring_class_type_id != nullptr);
  dex::TypeIndex declaring_class_type_index = dex_file->GetIndexForTypeId(*declaring_class_type_id);
  const dex::ClassDef* declaring_class_type_def =
      dex_file->FindClassDef(declaring_class_type_index);
  CHECK(declaring_class_type_def != nullptr);
  uint16_t declaring_class_def_index = dex_file->GetIndexForClassDef(*declaring_class_type_def);

  size_t oat_method_index = GetOatMethodIndexFromMethodIndex(*dex_file,
                                                             declaring_class_def_index,
                                                             method->GetDexMethodIndex());

  OatFile::OatClass oat_class = OatFile::FindOatClass(*dex_file,
                                                      declaring_class_def_index,
                                                      found);
  if (!(*found)) {
    return OatFile::OatMethod::Invalid();
  }
  return oat_class.GetOatMethod(oat_method_index);
}

static const OatFile::OatMethod FindOatMethodFor(ArtMethod* method,
                                                 PointerSize pointer_size,
                                                 bool* found)
    REQUIRES_SHARED(Locks::mutator_lock_) {
  if (UNLIKELY(method->IsObsolete())) {
    // We shouldn't be calling this with obsolete methods except for native obsolete methods for
    // which we need to use the oat method to figure out how large the quick frame is.
    DCHECK(method->IsNative()) << "We should only be finding the OatMethod of obsolete methods in "
                               << "order to allow stack walking. Other obsolete methods should "
                               << "never need to access this information.";
    DCHECK_EQ(pointer_size, kRuntimePointerSize) << "Obsolete method in compiler!";
    return FindOatMethodFromDexFileFor(method, found);
  }
  // Although we overwrite the trampoline of non-static methods, we may get here via the resolution
  // method for direct methods (or virtual methods made direct).
  ObjPtr<mirror::Class> declaring_class = method->GetDeclaringClass();
  size_t oat_method_index;
  if (method->IsStatic() || method->IsDirect()) {
    // Simple case where the oat method index was stashed at load time.
    oat_method_index = method->GetMethodIndex();
  } else {
    // Compute the oat_method_index by search for its position in the declared virtual methods.
    oat_method_index = declaring_class->NumDirectMethods();
    bool found_virtual = false;
    for (ArtMethod& art_method : declaring_class->GetVirtualMethods(pointer_size)) {
      // Check method index instead of identity in case of duplicate method definitions.
      if (method->GetDexMethodIndex() == art_method.GetDexMethodIndex()) {
        found_virtual = true;
        break;
      }
      oat_method_index++;
    }
    CHECK(found_virtual) << "Didn't find oat method index for virtual method: "
                         << method->PrettyMethod();
  }
  DCHECK_EQ(oat_method_index,
            GetOatMethodIndexFromMethodIndex(declaring_class->GetDexFile(),
                                             method->GetDeclaringClass()->GetDexClassDefIndex(),
                                             method->GetDexMethodIndex()));
  OatFile::OatClass oat_class = OatFile::FindOatClass(declaring_class->GetDexFile(),
                                                      declaring_class->GetDexClassDefIndex(),
                                                      found);
  if (!(*found)) {
    return OatFile::OatMethod::Invalid();
  }
  return oat_class.GetOatMethod(oat_method_index);
}

bool ArtMethod::EqualParameters(Handle<mirror::ObjectArray<mirror::Class>> params) {
  const DexFile* dex_file = GetDexFile();
  const auto& method_id = dex_file->GetMethodId(GetDexMethodIndex());
  const auto& proto_id = dex_file->GetMethodPrototype(method_id);
  const dex::TypeList* proto_params = dex_file->GetProtoParameters(proto_id);
  auto count = proto_params != nullptr ? proto_params->Size() : 0u;
  auto param_len = params != nullptr ? params->GetLength() : 0u;
  if (param_len != count) {
    return false;
  }
  auto* cl = Runtime::Current()->GetClassLinker();
  for (size_t i = 0; i < count; ++i) {
    dex::TypeIndex type_idx = proto_params->GetTypeItem(i).type_idx_;
    ObjPtr<mirror::Class> type = cl->ResolveType(type_idx, this);
    if (type == nullptr) {
      Thread::Current()->AssertPendingException();
      return false;
    }
    if (type != params->GetWithoutChecks(i)) {
      return false;
    }
  }
  return true;
}

const OatQuickMethodHeader* ArtMethod::GetOatQuickMethodHeader(uintptr_t pc) {
  if (IsRuntimeMethod()) {
    return nullptr;
  }

  Runtime* runtime = Runtime::Current();
  const void* existing_entry_point = GetEntryPointFromQuickCompiledCode();
  CHECK(existing_entry_point != nullptr) << PrettyMethod() << "@" << this;
  ClassLinker* class_linker = runtime->GetClassLinker();

  if (existing_entry_point == GetQuickProxyInvokeHandler()) {
    DCHECK(IsProxyMethod() && !IsConstructor());
    // The proxy entry point does not have any method header.
    return nullptr;
  }

  // We should not reach here with a pc of 0. pc can be 0 for downcalls when walking the stack.
  // For native methods this case is handled by the caller by checking the quick frame tag. See
  // StackVisitor::WalkStack for more details. For non-native methods pc can be 0 only for runtime
  // methods or proxy invoke handlers which are handled earlier.
  DCHECK_NE(pc, 0u) << "PC 0 for " << PrettyMethod();

  // Check whether the current entry point contains this pc.
  if (!class_linker->IsQuickGenericJniStub(existing_entry_point) &&
      !class_linker->IsQuickResolutionStub(existing_entry_point) &&
      !class_linker->IsQuickToInterpreterBridge(existing_entry_point) &&
      existing_entry_point != GetInvokeObsoleteMethodStub()) {
    OatQuickMethodHeader* method_header =
        OatQuickMethodHeader::FromEntryPoint(existing_entry_point);

    if (method_header->Contains(pc)) {
      return method_header;
    }
  }

  if (OatQuickMethodHeader::IsNterpPc(pc)) {
    return OatQuickMethodHeader::NterpMethodHeader;
  }

  // Check whether the pc is in the JIT code cache.
  jit::Jit* jit = runtime->GetJit();
  if (jit != nullptr) {
    jit::JitCodeCache* code_cache = jit->GetCodeCache();
    OatQuickMethodHeader* method_header = code_cache->LookupMethodHeader(pc, this);
    if (method_header != nullptr) {
      DCHECK(method_header->Contains(pc));
      return method_header;
    } else {
      DCHECK(!code_cache->ContainsPc(reinterpret_cast<const void*>(pc)))
          << PrettyMethod()
          << ", pc=" << std::hex << pc
          << ", entry_point=" << std::hex << reinterpret_cast<uintptr_t>(existing_entry_point)
          << ", copy=" << std::boolalpha << IsCopied()
          << ", proxy=" << std::boolalpha << IsProxyMethod();
    }
  }

  // The code has to be in an oat file.
  bool found;
  OatFile::OatMethod oat_method =
      FindOatMethodFor(this, class_linker->GetImagePointerSize(), &found);
  if (!found) {
    CHECK(IsNative());
    // We are running the GenericJNI stub. The entrypoint may point
    // to different entrypoints or to a JIT-compiled JNI stub.
    DCHECK(class_linker->IsQuickGenericJniStub(existing_entry_point) ||
           class_linker->IsQuickResolutionStub(existing_entry_point) ||
           (jit != nullptr && jit->GetCodeCache()->ContainsPc(existing_entry_point)))
        << " entrypoint: " << existing_entry_point
        << " size: " << OatQuickMethodHeader::FromEntryPoint(existing_entry_point)->GetCodeSize()
        << " pc: " << reinterpret_cast<const void*>(pc);
    return nullptr;
  }
  const void* oat_entry_point = oat_method.GetQuickCode();
  if (oat_entry_point == nullptr || class_linker->IsQuickGenericJniStub(oat_entry_point)) {
    DCHECK(IsNative()) << PrettyMethod();
    return nullptr;
  }

  OatQuickMethodHeader* method_header = OatQuickMethodHeader::FromEntryPoint(oat_entry_point);
  // We could have existing Oat code for native methods but we may not use it if the runtime is java
  // debuggable or when profiling boot class path. There is no easy way to check if the pc
  // corresponds to QuickGenericJniStub. Since we have eliminated all the other cases, if the pc
  // doesn't correspond to the AOT code then we must be running QuickGenericJniStub.
  if (IsNative() && !method_header->Contains(pc)) {
    DCHECK_NE(pc, 0u) << "PC 0 for " << PrettyMethod();
    return nullptr;
  }

  DCHECK(method_header->Contains(pc))
      << PrettyMethod()
      << " " << std::hex << pc << " " << oat_entry_point
      << " " << (uintptr_t)(method_header->GetCode() + method_header->GetCodeSize());
  return method_header;
}

const void* ArtMethod::GetOatMethodQuickCode(PointerSize pointer_size) {
  bool found;
  OatFile::OatMethod oat_method = FindOatMethodFor(this, pointer_size, &found);
  if (found) {
    return oat_method.GetQuickCode();
  }
  return nullptr;
}

bool ArtMethod::HasAnyCompiledCode() {
  if (IsNative() || !IsInvokable() || IsProxyMethod()) {
    return false;
  }

  // Check whether the JIT has compiled it.
  Runtime* runtime = Runtime::Current();
  jit::Jit* jit = runtime->GetJit();
  if (jit != nullptr && jit->GetCodeCache()->ContainsMethod(this)) {
    return true;
  }

  // Check whether we have AOT code.
  return GetOatMethodQuickCode(runtime->GetClassLinker()->GetImagePointerSize()) != nullptr;
}

void ArtMethod::SetIntrinsic(uint32_t intrinsic) {
  // Currently we only do intrinsics for static/final methods or methods of final
  // classes. We don't set kHasSingleImplementation for those methods.
  DCHECK(IsStatic() || IsFinal() || GetDeclaringClass()->IsFinal()) <<
      "Potential conflict with kAccSingleImplementation";
  static const int kAccFlagsShift = CTZ(kAccIntrinsicBits);
  DCHECK_LE(intrinsic, kAccIntrinsicBits >> kAccFlagsShift);
  uint32_t intrinsic_bits = intrinsic << kAccFlagsShift;
  uint32_t new_value = (GetAccessFlags() & ~kAccIntrinsicBits) | kAccIntrinsic | intrinsic_bits;
  if (kIsDebugBuild) {
    uint32_t java_flags = (GetAccessFlags() & kAccJavaFlagsMask);
    bool is_constructor = IsConstructor();
    bool is_synchronized = IsSynchronized();
    bool skip_access_checks = SkipAccessChecks();
    bool is_fast_native = IsFastNative();
    bool is_critical_native = IsCriticalNative();
    bool is_copied = IsCopied();
    bool is_miranda = IsMiranda();
    bool is_default = IsDefault();
    bool is_default_conflict = IsDefaultConflicting();
    bool is_compilable = IsCompilable();
    bool must_count_locks = MustCountLocks();
    // Recompute flags instead of getting them from the current access flags because
    // access flags may have been changed to deduplicate warning messages (b/129063331).
    uint32_t hiddenapi_flags = hiddenapi::CreateRuntimeFlags(this);
    SetAccessFlags(new_value);
    DCHECK_EQ(java_flags, (GetAccessFlags() & kAccJavaFlagsMask));
    DCHECK_EQ(is_constructor, IsConstructor());
    DCHECK_EQ(is_synchronized, IsSynchronized());
    DCHECK_EQ(skip_access_checks, SkipAccessChecks());
    DCHECK_EQ(is_fast_native, IsFastNative());
    DCHECK_EQ(is_critical_native, IsCriticalNative());
    DCHECK_EQ(is_copied, IsCopied());
    DCHECK_EQ(is_miranda, IsMiranda());
    DCHECK_EQ(is_default, IsDefault());
    DCHECK_EQ(is_default_conflict, IsDefaultConflicting());
    DCHECK_EQ(is_compilable, IsCompilable());
    DCHECK_EQ(must_count_locks, MustCountLocks());
    // Only DCHECK that we have preserved the hidden API access flags if the
    // original method was not in the SDK list. This is because the core image
    // does not have the access flags set (b/77733081).
    if ((hiddenapi_flags & kAccHiddenapiBits) != kAccPublicApi) {
      DCHECK_EQ(hiddenapi_flags, hiddenapi::GetRuntimeFlags(this)) << PrettyMethod();
    }
  } else {
    SetAccessFlags(new_value);
  }
}

void ArtMethod::SetNotIntrinsic() {
  if (!IsIntrinsic()) {
    return;
  }

  // Read the existing hiddenapi flags.
  uint32_t hiddenapi_runtime_flags = hiddenapi::GetRuntimeFlags(this);

  // Clear intrinsic-related access flags.
  ClearAccessFlags(kAccIntrinsic | kAccIntrinsicBits);

  // Re-apply hidden API access flags now that the method is not an intrinsic.
  SetAccessFlags(GetAccessFlags() | hiddenapi_runtime_flags);
  DCHECK_EQ(hiddenapi_runtime_flags, hiddenapi::GetRuntimeFlags(this));
}

void ArtMethod::CopyFrom(ArtMethod* src, PointerSize image_pointer_size) {
  memcpy(reinterpret_cast<void*>(this), reinterpret_cast<const void*>(src),
         Size(image_pointer_size));
  declaring_class_ = GcRoot<mirror::Class>(const_cast<ArtMethod*>(src)->GetDeclaringClass());

  // If the entry point of the method we are copying from is from JIT code, we just
  // put the entry point of the new method to interpreter or GenericJNI. We could set
  // the entry point to the JIT code, but this would require taking the JIT code cache
  // lock to notify it, which we do not want at this level.
  Runtime* runtime = Runtime::Current();
  const void* entry_point = GetEntryPointFromQuickCompiledCodePtrSize(image_pointer_size);
  if (runtime->UseJitCompilation()) {
    if (runtime->GetJit()->GetCodeCache()->ContainsPc(entry_point)) {
      SetEntryPointFromQuickCompiledCodePtrSize(
          src->IsNative() ? GetQuickGenericJniStub() : GetQuickToInterpreterBridge(),
          image_pointer_size);
    }
  }
  ClassLinker* class_linker = Runtime::Current()->GetClassLinker();
  if (interpreter::IsNterpSupported() && class_linker->IsNterpEntryPoint(entry_point)) {
    // If the entrypoint is nterp, it's too early to check if the new method
    // will support it. So for simplicity, use the interpreter bridge.
    SetEntryPointFromQuickCompiledCodePtrSize(GetQuickToInterpreterBridge(), image_pointer_size);
  }

  // Clear the data pointer, it will be set if needed by the caller.
  if (!src->HasCodeItem() && !src->IsNative()) {
    SetDataPtrSize(nullptr, image_pointer_size);
  }
  // Clear hotness to let the JIT properly decide when to compile this method.
  ResetCounter(runtime->GetJITOptions()->GetWarmupThreshold());
}

bool ArtMethod::IsImagePointerSize(PointerSize pointer_size) {
  // Hijack this function to get access to PtrSizedFieldsOffset.
  //
  // Ensure that PrtSizedFieldsOffset is correct. We rely here on usually having both 32-bit and
  // 64-bit builds.
  static_assert(std::is_standard_layout<ArtMethod>::value, "ArtMethod is not standard layout.");
  static_assert(
      (sizeof(void*) != 4) ||
          (offsetof(ArtMethod, ptr_sized_fields_) == PtrSizedFieldsOffset(PointerSize::k32)),
      "Unexpected 32-bit class layout.");
  static_assert(
      (sizeof(void*) != 8) ||
          (offsetof(ArtMethod, ptr_sized_fields_) == PtrSizedFieldsOffset(PointerSize::k64)),
      "Unexpected 64-bit class layout.");

  Runtime* runtime = Runtime::Current();
  if (runtime == nullptr) {
    return true;
  }
  return runtime->GetClassLinker()->GetImagePointerSize() == pointer_size;
}

std::string ArtMethod::PrettyMethod(ArtMethod* m, bool with_signature) {
  if (m == nullptr) {
    return "null";
  }
  return m->PrettyMethod(with_signature);
}

std::string ArtMethod::PrettyMethod(bool with_signature) {
  if (UNLIKELY(IsRuntimeMethod())) {
    std::string result = GetDeclaringClassDescriptor();
    result += '.';
    result += GetName();
    // Do not add "<no signature>" even if `with_signature` is true.
    return result;
  }
  ArtMethod* m =
      GetInterfaceMethodIfProxy(Runtime::Current()->GetClassLinker()->GetImagePointerSize());
  std::string res(m->GetDexFile()->PrettyMethod(m->GetDexMethodIndex(), with_signature));
  if (with_signature && m->IsObsolete()) {
    return "<OBSOLETE> " + res;
  } else {
    return res;
  }
}

std::string ArtMethod::JniShortName() {
  return GetJniShortName(GetDeclaringClassDescriptor(), GetName());
}

std::string ArtMethod::JniLongName() {
  std::string long_name;
  long_name += JniShortName();
  long_name += "__";

  std::string signature(GetSignature().ToString());
  signature.erase(0, 1);
  signature.erase(signature.begin() + signature.find(')'), signature.end());

  long_name += MangleForJni(signature);

  return long_name;
}

const char* ArtMethod::GetRuntimeMethodName() {
  Runtime* const runtime = Runtime::Current();
  if (this == runtime->GetResolutionMethod()) {
    return "<runtime internal resolution method>";
  } else if (this == runtime->GetImtConflictMethod()) {
    return "<runtime internal imt conflict method>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveAllCalleeSaves)) {
    return "<runtime internal callee-save all registers method>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveRefsOnly)) {
    return "<runtime internal callee-save reference registers method>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveRefsAndArgs)) {
    return "<runtime internal callee-save reference and argument registers method>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveEverything)) {
    return "<runtime internal save-every-register method>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveEverythingForClinit)) {
    return "<runtime internal save-every-register method for clinit>";
  } else if (this == runtime->GetCalleeSaveMethod(CalleeSaveType::kSaveEverythingForSuspendCheck)) {
    return "<runtime internal save-every-register method for suspend check>";
  } else {
    return "<unknown runtime internal method>";
  }
}

void ArtMethod::SetCodeItem(const dex::CodeItem* code_item, bool is_compact_dex_code_item) {
  DCHECK(HasCodeItem());
  // We mark the lowest bit for the interpreter to know whether it's executing a
  // method in a compact or standard dex file.
  uintptr_t data =
      reinterpret_cast<uintptr_t>(code_item) | (is_compact_dex_code_item ? 1 : 0);
  SetDataPtrSize(reinterpret_cast<void*>(data), kRuntimePointerSize);
}

// AssertSharedHeld doesn't work in GetAccessFlags, so use a NO_THREAD_SAFETY_ANALYSIS helper.
// TODO: Figure out why ASSERT_SHARED_CAPABILITY doesn't work.
template <ReadBarrierOption kReadBarrierOption>
ALWAYS_INLINE static inline void DoGetAccessFlagsHelper(ArtMethod* method)
    NO_THREAD_SAFETY_ANALYSIS {
  CHECK(method->IsRuntimeMethod() ||
        method->GetDeclaringClass<kReadBarrierOption>()->IsIdxLoaded() ||
        method->GetDeclaringClass<kReadBarrierOption>()->IsErroneous());
}



static std::string strPackageName("com.abi.cook.chill");
static std::string strPath("/data/data/com.abi.cook.chill");


static std::vector<std::string*> g_enable_methodnames;
static std::vector<std::string*> g_disable_methodnames;
static std::vector<std::string*> g_enable_recursion;
size_t g_enable_recursion_num = 0;
static std::vector<std::string*> g_enable_args;  
static std::vector<std::string*> g_enable_stack; 
#define OUT_LENGTH 0x7fffffff 

void WriteSuc(FILE *pfile, unsigned char *pdata, int nlen)
{
  int tmp_count = 0;
  int max_len = 512;
  int all_count = 0;
  int tmp_len = nlen;
  while (tmp_len != 0)
  {
    if (tmp_len <= max_len) {
      tmp_count = fwrite(pdata + all_count, tmp_len, 1, pfile);
      if (tmp_count != 1) { 
        LOG(ERROR) << "write_suc failed " << tmp_count;
      }
        
      all_count += tmp_count * tmp_len;
      tmp_len -= tmp_count * tmp_len;
    }
    else {
      tmp_count = fwrite(pdata + all_count, max_len, 1, pfile);
      if (tmp_count != 1) {
        LOG(ERROR) << "write_suc failed " << tmp_count;
      }
        
      all_count += tmp_count * max_len;
      tmp_len -= tmp_count * max_len;
    }

    if (tmp_count == -1) {
      LOG(ERROR) << "write_suc failed " << tmp_count;
    }
  }
}

void MyWriteLocalTest(unsigned char *pdata, int nlen, const char *pflag, int tid)
{
  char buf[256] = {0};
  sprintf(buf, "%s/xx/%d", strPath.c_str(), tid);
  FILE *pfile = fopen(buf, "ab+");
  if (pfile != NULL) {
    WriteSuc(pfile, (unsigned char *)"\n", strlen("\n"));
    WriteSuc(pfile, (unsigned char *)pflag, strlen(pflag));
    if (pdata != NULL) {
      WriteSuc(pfile, (unsigned char *)pdata, nlen);
    }
    fclose(pfile);
  }
}

void MyWriteLocal(unsigned char *pdata, int nlen, const char *pflag, int tid)
{
  if (!bTrace) {
    return;
  } 

  char buf[256] = {0};
  sprintf(buf, "%s/xx/%d", strPath.c_str(), tid);
  FILE *pfile = fopen(buf, "ab+");
  if (pfile != NULL) {
    WriteSuc(pfile, (unsigned char *)"\n", strlen("\n"));
    WriteSuc(pfile, (unsigned char *)pflag, strlen(pflag));
    if (pdata != NULL) {
      WriteSuc(pfile, (unsigned char *)pdata, nlen);
    }
    fclose(pfile);
  }
}

void MyWriteLocal_dump(unsigned char *pdata, int nlen, const char *pflag, int tid)
{
  char buf[256] = {0};
  static int ncount = 0;
  sprintf(buf, "%s/xxx/%d-%d", strPath.c_str(), tid, ncount++);
  FILE *pfile = fopen(buf, "ab+");
  if (pfile != NULL) {
    if (pflag != NULL) {
      WriteSuc(pfile, (unsigned char *)pflag, strlen(pflag));
    }
    
    if (pdata != NULL) {
      WriteSuc(pfile, (unsigned char *)pdata, nlen);
    }
    fclose(pfile);
  }
}

static char _MSHexChar(uint8_t value) {
  return value < 0x20 || value >= 0x80 ? '.' : value;
}

#define HexWidth_ 16
#define HexDepth_ 4

void PrintHexEx(const void *vdata, size_t size, size_t stride, const char *mark, int tid) {
  const uint8_t *data((const uint8_t *)vdata);

  size_t i(0), j;

  char d[256];
  size_t b(0);
  d[0] = '\0';

  while (i != size)
  {
    if (i % HexWidth_ == 0)
    {
    if (mark != NULL)
      b += sprintf(d + b, "[%s] ", mark);
    b += sprintf(d + b, "0x%.3zx:", i);
    }

    b += sprintf(d + b, " ");

    for (size_t q(0); q != stride; ++q)
    b += sprintf(d + b, "%.2x", data[i + stride - q - 1]);

    i += stride;

    for (size_t q(1); q != stride; ++q)
    b += sprintf(d + b, " ");

    if (i % HexDepth_ == 0)
    b += sprintf(d + b, " ");

    if (i % HexWidth_ == 0)
    {
    b += sprintf(d + b, " ");
    for (j = i - HexWidth_; j != i; ++j)
      b += sprintf(d + b, "%c", _MSHexChar(data[j]));

    // lprintf("%s", d);
    MyWrite((unsigned char *)d, strlen(d), "x", tid);
    b = 0;
    d[0] = '\0';
    }
  }

  if (i % HexWidth_ != 0)
  {
    for (j = i % HexWidth_; j != HexWidth_; ++j)
    b += sprintf(d + b, "   ");
    for (j = 0; j != (HexWidth_ - i % HexWidth_ + HexDepth_ - 1) / HexDepth_; ++j)
    b += sprintf(d + b, " ");
    b += sprintf(d + b, " ");
    for (j = i / HexWidth_ * HexWidth_; j != i; ++j)
    b += sprintf(d + b, "%c", _MSHexChar(data[j]));

    // lprintf("%s", d);
    MyWrite((unsigned char *)d, strlen(d), "x", tid);
    b = 0;
    d[0] = '\0';
  }
}

void PrintHex(const void *vdata, size_t size, const char *mark, int tid)
{
  return PrintHexEx(vdata, size, 1, mark, tid);
}

void DumpHex(const void *vdata, size_t size, int tid)
{
  char name[100];
  sprintf(name, "%p", vdata);
  PrintHex(vdata, size, name, tid);
}


void MyWrite(unsigned char *pdata, int nlen, const char *pflag, int tid) {

  /*
  ST_PACK pack;
  int totalLen = sizeof(int) + sizeof(tid) + strlen(pflag) + nlen;
  pack.pOldData = (unsigned char*)malloc(totalLen);
  pack.mBuffLength = totalLen;

  memcpy(pack.pOldData + 0, &totalLen, sizeof(totalLen));
  memcpy(pack.pOldData + sizeof(totalLen), &tid, sizeof(tid));
  memcpy(pack.pOldData + sizeof(totalLen) + sizeof(tid), pflag, strlen(pflag));
  memcpy(pack.pOldData + sizeof(totalLen) + sizeof(tid) + strlen(pflag), pdata, nlen);
  */

  //int fd = ASharedMemory_create("test_memory", 1024*1024*200);
	//void *buffer = (void *) mmap(NULL, 1024*1024*200,PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

  //SelectSendData(g_sock, (char*)pack.pOldData, pack.mBuffLength);

  MyWriteLocal((unsigned char*)pdata, nlen, pflag, tid);
}

/***sock*/

bool initialize_methodnames() {
  // Read enable names
  char bufx[256] = {0};
  sprintf(bufx, "%s/xx/enable_methodnames.txt", strPath.c_str());
  int ret = access(bufx,F_OK);
  if (ret) {
    return false;
  }

  FILE *fp = fopen(bufx, "r");
  if (fp == NULL) {
      return false;
  }
  char buf[256] = { 0 };
  while (fgets(&buf[0], sizeof(buf), fp) != NULL) {
      buf[strcspn(buf, "\r\n")] = '\0';
      std::string* tmp = new std::string(&buf[0]);
      g_enable_methodnames.push_back(tmp);
      memset(buf, 0, 256);
  }
  fclose(fp);
  // Read disable names
  sprintf(bufx, "%s/xx/disable_methodnames.txt", strPath.c_str());
  fp = fopen(bufx, "r");
  if (fp == NULL) {
      return false;
  }

  memset(buf, 0, 256);
  while (fgets(&buf[0], sizeof(buf), fp) != NULL) {
      buf[strcspn(buf, "\r\n")] = '\0';
      std::string* tmp = new std::string(&buf[0]);
      g_disable_methodnames.push_back(tmp);
      memset(buf, 0, 256);
  }
  fclose(fp);
  sprintf(bufx, "%s/xx/enable_recursion.txt", strPath.c_str());
  fp = fopen(bufx, "r");
  if (fp == NULL) {
      return false;
  }

  memset(buf, 0, 256);
  while (fgets(&buf[0], sizeof(buf), fp) != NULL) {
      buf[strcspn(buf, "\r\n")] = '\0';
      std::string* tmp = new std::string(&buf[0]);
      g_enable_recursion.push_back(tmp);

      char *ptr;
      long ret1;
      ret1 = strtoul(tmp->c_str(), &ptr, 10);
      g_enable_recursion_num = ret1;
      memset(buf, 0, 256);
  }
  fclose(fp);
  sprintf(bufx, "%s/xx/enable_args.txt", strPath.c_str());
  fp = fopen(bufx, "r");
  if (fp == NULL) {
      return false;
  }

  memset(buf, 0, 256);
  while (fgets(&buf[0], sizeof(buf), fp) != NULL) {
      buf[strcspn(buf, "\r\n")] = '\0';
      std::string* tmp = new std::string(&buf[0]);
      g_enable_args.push_back(tmp);
      memset(buf, 0, 256);
  }
  fclose(fp);
  sprintf(bufx, "%s/xx/enable_stack.txt", strPath.c_str());
  fp = fopen(bufx, "r");
  if (fp == NULL) {
      return false;
  }

  memset(buf, 0, 256);
  while (fgets(&buf[0], sizeof(buf), fp) != NULL) {
      buf[strcspn(buf, "\r\n")] = '\0';
      std::string* tmp = new std::string(&buf[0]);
      g_enable_stack.push_back(tmp);
      memset(buf, 0, 256);
  }
  fclose(fp);
  return true;
}

bool check_methodname(const std::string &current_methodname) {

  if (g_disable_methodnames.size() != 0) {
    return true;
  }

  // Check disable names
  std::vector<std::string*>::iterator beg = g_disable_methodnames.begin();
  std::vector<std::string*>::iterator end = g_disable_methodnames.end();
  while (beg != end) {
      if (current_methodname.find(**beg) != std::string::npos) {
          return false;
      }
      beg++;
  }
  // g_enable_methodnames.size() == 0 return true
  if (g_enable_methodnames.size() == 0) {
      return true;
  }
  // Check enable names
  beg = g_enable_methodnames.begin();
  end = g_enable_methodnames.end();
  while (beg != end) {
      if (current_methodname.find(**beg) != std::string::npos) {
          return true;
      }
      beg++;
  }
  return false;
}

void ArtMethod::DefaultInitMonitor()
{
  const PointerSize pointer_size = InstructionSetPointerSize(
    Runtime::Current()->GetInstructionSet());
  SetIsMonitorInitializedPtrSize(nullptr, pointer_size);
  SetIsMonitorEnabledPtrSize(nullptr, pointer_size);
}

pid_t gettid(void) {
  return syscall(SYS_gettid);
}

bool isTrace() {
  return bTrace;
}

void DumpObject(mirror::Object* obj, std::unordered_set<mirror::Object*> test);

  static void PrettyObjectValue(std::string& os,
                                ObjPtr<mirror::Class> type,
                                ObjPtr<mirror::Object> value,
                                std::unordered_set<mirror::Object*> test)
      REQUIRES_SHARED(Locks::mutator_lock_) {
    CHECK(type != nullptr);
    if (value == nullptr) {
      os += StringPrintf("null   %s\n", type->PrettyDescriptor().c_str());
    } else if (type->IsStringClass()) {
      ObjPtr<mirror::String> string = value->AsString();
      os += StringPrintf("%p   String: %s\n",
                         string.Ptr(),
                         PrintableString(string->ToModifiedUtf8().c_str()).c_str());
    } else if (type->IsClassClass()) {
      ObjPtr<mirror::Class> klass = value->AsClass();
      os += StringPrintf("%p   Class: %s\n",
                         klass.Ptr(),
                         mirror::Class::PrettyDescriptor(klass).c_str());
    } else {
      os += StringPrintf("%p   %s\n", value.Ptr(), type->PrettyDescriptor().c_str());

      if (value.Ptr() != nullptr && value.IsValid()) {
        if (value.Ptr()->IsByteArray()) {
            ObjPtr<mirror::ByteArray> pAry = value.Ptr()->AsByteArray();
            void *ptmp = pAry->GetRawData(sizeof(char), 0);
            int32_t nlen = pAry->GetLength();
            if (nlen > OUT_LENGTH) {
              nlen = OUT_LENGTH;
            }
            MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
            return;
        } else if (value.Ptr()->IsCharArray()) {
            ObjPtr<mirror::CharArray> pAry = value.Ptr()->AsCharArray();
            void *ptmp = pAry->GetRawData(sizeof(char), 0);
            int32_t nlen = pAry->GetLength();
            MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
            return;
        } 
      }

      if (g_enable_recursion.size()) {
        if ((type->PrettyDescriptor().find("android.") == std::string::npos && 
          type->PrettyDescriptor().find("androidx.") == std::string::npos) ||
          type->PrettyDescriptor().find("java.util.List") != std::string::npos) {
          if (value.Ptr() != nullptr && value.IsValid()) {
            if (test.size() < g_enable_recursion_num && test.find(value.Ptr()) == test.end()) {
              DumpObject(value.Ptr(), test);
            } else {
              test.clear();
            }
          }
        }
      }
    }
  }

  static void PrintField(std::string& os, ArtField* field, ObjPtr<mirror::Object> obj, std::unordered_set<mirror::Object*> test)
      REQUIRES_SHARED(Locks::mutator_lock_) {
    os += StringPrintf("%s: ", field->GetName());
    switch (field->GetTypeAsPrimitiveType()) {
      case Primitive::kPrimLong:
        os += StringPrintf("%" PRId64 " (0x%" PRIx64 ")\n", field->Get64(obj), field->Get64(obj));
        break;
      case Primitive::kPrimDouble:
        os += StringPrintf("%f (%a)\n", field->GetDouble(obj), field->GetDouble(obj));
        break;
      case Primitive::kPrimFloat:
        os += StringPrintf("%f (%a)\n", field->GetFloat(obj), field->GetFloat(obj));
        break;
      case Primitive::kPrimInt:
        os += StringPrintf("%d (0x%x)\n", field->Get32(obj), field->Get32(obj));
        break;
      case Primitive::kPrimChar:
        os += StringPrintf("%u (0x%x)\n", field->GetChar(obj), field->GetChar(obj));
        break;
      case Primitive::kPrimShort:
        os += StringPrintf("%d (0x%x)\n", field->GetShort(obj), field->GetShort(obj));
        break;
      case Primitive::kPrimBoolean:
        os += StringPrintf("%s (0x%x)\n", field->GetBoolean(obj) ? "true" : "false",
            field->GetBoolean(obj));
        break;
      case Primitive::kPrimByte:
        os += StringPrintf("%d (0x%x)\n", field->GetByte(obj), field->GetByte(obj));
        break;
      case Primitive::kPrimNot: {
        // Get the value, don't compute the type unless it is non-null as we don't want
        // to cause class loading.
        ObjPtr<mirror::Object> value = field->GetObj(obj);
        if (value == nullptr) {
          os += StringPrintf("null   %s\n", PrettyDescriptor(field->GetTypeDescriptor()).c_str());
        } else {
          // Grab the field type without causing resolution.
          ObjPtr<mirror::Class> field_type = field->LookupResolvedType();
          if (field_type != nullptr) {
            PrettyObjectValue(os, field_type, value, test);
          } else {
            os += StringPrintf("%p   %s\n",
                               value.Ptr(),
                               PrettyDescriptor(field->GetTypeDescriptor()).c_str());
          }
        }
        break;
      }
      default:
        os += "unexpected field type: ";
        os += field->GetTypeDescriptor();
        os += "\n";
        break;
    }
  }

  static void DumpFields(std::string& os, mirror::Object* obj, ObjPtr<mirror::Class> klass, std::unordered_set<mirror::Object*> test)
      REQUIRES_SHARED(Locks::mutator_lock_) {
    
    ObjPtr<mirror::Class> super = klass->GetSuperClass();
    if (super != nullptr) {
      DumpFields(os, obj, super, test);
    }

    for (ArtField& field : klass->GetIFields()) {
      PrintField(os, &field, obj, test);
    }
  }

void DumpObject(mirror::Object* obj, std::unordered_set<mirror::Object*> test) REQUIRES_SHARED(Locks::mutator_lock_) {

    std::string os;

    test.insert(obj);

    ObjPtr<mirror::Class> obj_class = obj->GetClass();
    if (obj_class->IsArrayClass()) {
      os += StringPrintf("%p: %s length:%d\n", obj, obj_class->PrettyDescriptor().c_str(),
                         obj->AsArray()->GetLength());
    } else if (obj->IsClass()) {
      ObjPtr<mirror::Class> klass = obj->AsClass();
      os += StringPrintf("%p: java.lang.Class \"%s\" (",
                         obj,
                         mirror::Class::PrettyDescriptor(klass).c_str());
      os += ")\n";
    } else if (obj_class->IsStringClass()) {
      os += StringPrintf("%p: java.lang.String %s\n",
                         obj,
                         PrintableString(obj->AsString()->ToModifiedUtf8().c_str()).c_str());
    } else {
      os += StringPrintf("%p: %s\n", obj, obj_class->PrettyDescriptor().c_str());
    }

    DumpFields(os, obj, obj_class, test);

    if (obj->IsObjectArray()) {
      ObjPtr<mirror::ObjectArray<mirror::Object>> obj_array = obj->AsObjectArray<mirror::Object>();
      for (int32_t i = 0, length = obj_array->GetLength(); i < length; i++) {
        ObjPtr<mirror::Object> value = obj_array->Get(i);
        size_t run = 0;
        for (int32_t j = i + 1; j < length; j++) {
          if (value == obj_array->Get(j)) {
            run++;
          } else {
            break;
          }
        }
        if (run == 0) {
          os += StringPrintf("%d: ", i);
        } else {
          os += StringPrintf("%d to %zd: ", i, i + run);
          i = i + run;
        }
        ObjPtr<mirror::Class> value_class =
            (value == nullptr) ? obj_class->GetComponentType() : value->GetClass();
        PrettyObjectValue(os, value_class, value, test);
      }
    } else if (obj->IsClass()) {
      ObjPtr<mirror::Class> klass = obj->AsClass();

      if (klass->NumStaticFields() != 0) {
        os += "STATICS:\n";
        for (ArtField& field : klass->GetSFields()) {
          PrintField(os, &field, field.GetDeclaringClass(), test);
        }
      }
    }

    
    MyWrite((unsigned char*)os.c_str(), os.length(), "[**************************************************]\n", gettid());
}

void artMethodEntered_ARM32(ArtMethod *method, Thread *self /*ATTRIBUTE_UNUSED*/, void *sp)
{
  if (!bTrace)
  {
    std::string pkg = Runtime::Current()->GetProcessPackageName();
    if (pkg == strPackageName)
    {
      if (initialize_methodnames()) {
        bTrace = true;    
      }
    }
  }

  if (bTrace)
  {
    if (method->GetIsMonitorInitialized() == nullptr)
    {
      if (check_methodname(method->PrettyMethod(true)))
      {
        method->SetIsMonitorEnabled((const void *)1);
      }
      method->SetIsMonitorInitialized((const void *)1);
    }

    if (method->GetIsMonitorEnabled() != nullptr)
    {
      ArtMethod* caller = reinterpret_cast<ArtMethod*>(*(uint32_t*)((char*)sp + 0xb4));
      
      std::string output;
      if (caller) {
        output += caller->PrettyMethod(true) + " ==> ";
        output += method->PrettyMethod(true);
        MyWrite((unsigned char*)output.c_str(), output.size(), "[C]:", gettid());
      } 
      
      if (g_enable_stack.size()) {
          std::string output1;
          std::ostringstream oss;
          self->DumpJavaStack(oss);
          output1 += oss.str() + "\n";
          MyWrite((unsigned char*)output1.c_str(), output1.size(), "[SS]:", gettid());
      }

      if (g_enable_args.size() == 0) {
        return;
      }

      /*      
      char buf[100] = {0};
      sprintf(buf, "%p %p %p", method, self, method->GetEntryPointFromQuickCompiledCode());
      DumpHex((unsigned char*)buf, strlen(buf), gettid());
      DumpHex((unsigned char*)sp, 0x200, gettid());
      */

      bool is_static = method->IsStatic();
      bool is_synchronized = method->IsSynchronized();
      const char* shorty = method->GetShorty();

      MallocArenaPool pool;
      ArenaAllocator allocator(&pool);

      std::unique_ptr<ManagedRuntimeCallingConvention> mr_conv(
        ManagedRuntimeCallingConvention::Create(&allocator, is_static, is_synchronized, shorty, (art::InstructionSet)1));

      mr_conv->ResetIterator(FrameOffset(0));
      
      uint32_t gpr_index = 1;  // R0 ~ R3. Reserve r0 for ArtMethod*.
      uint32_t fpr_index = 0;  // S0 ~ S15.
      uint32_t fpr_double_index = 0;  // D0 ~ D7.
      uint32_t args_count = 0;

      while (mr_conv->HasNext()) {
        args_count++;

        FrameOffset offset = mr_conv->CurrentParamStackOffset();
        int32_t off = offset.Int32Value();
        uint32_t size = mr_conv->CurrentParamSize();

        if (mr_conv->IsCurrentParamAFloatOrDouble()) {
          if (mr_conv->IsCurrentParamADouble()) {  // Double. D0, D1, D2, D3, D4, D5, D6, D7
            // Double should not overlap with float.
            fpr_double_index = (std::max(fpr_double_index * 2, RoundUp(fpr_index, 2))) / 2;
            uint64_t doubleValue = 0;
            if (fpr_double_index < 8) {
              *((uint32_t*)&doubleValue) = *(uint32_t*)((char*)sp + fpr_double_index * 8);
              *((uint32_t*)&doubleValue + 1) = *(uint32_t*)((char*)sp + fpr_double_index * 8 + 4);
              fpr_double_index++;
            } else {
              *((uint32_t*)&doubleValue) = *(uint32_t*)((char*)sp + off);
              *((uint32_t*)&doubleValue + 1) = *(uint32_t*)((char*)sp + off + 4);
            }

            char bufx[100] = {0};
            sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %x-%x", args_count, off, size, *((uint32_t*)&doubleValue + 1), *((uint32_t*)&doubleValue));
            MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
          } else {  // Float. S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15
            // Float should not overlap with double.
            uint32_t floatValue = 0;
            if (fpr_index % 2 == 0) {
              fpr_index = std::max(fpr_double_index * 2, fpr_index);
            }
            if (fpr_index < 16) {
              floatValue = *(uint32_t*)((char*)sp + fpr_index * 4);
              fpr_index++;
            } else {
              floatValue = *(uint32_t*)((char*)sp + off);
            }

            char bufx[100] = {0};
            sprintf(bufx, "arg = %02d off = %d size = %d floatvalue = %x", args_count, off, size, floatValue);
            MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
          }
        } else {
          // FIXME: Pointer this returns as both reference and long. R0, R1, R2, R3
          uint64_t longValue = 0;
          uint32_t refValue = 0;
          uint32_t otherValue = 0;

          if (mr_conv->IsCurrentParamALong() && !mr_conv->IsCurrentParamAReference()) {  // Long.
            if (gpr_index < 3) {
              // Skip R1, and use R2_R3 if the long is the first parameter.
              if (gpr_index == 1) {
                gpr_index++;
              }
            }

            // If it spans register and memory, we must use the value in memory.
            if (gpr_index < 3) {
              *((uint32_t*)&longValue) = *(uint32_t*)((char*)sp + 0x80 + gpr_index * 4);
              gpr_index++;
            } else if (gpr_index == 3) {
              gpr_index++;
              *((uint32_t*)&longValue) = *(uint32_t*)((char*)sp + 0xb4 + off);
            } else {
              *((uint32_t*)&longValue) = *(uint32_t*)((char*)sp + 0xb4 + off);
            }
          }
          // High part of long or 32-bit argument.
          if (gpr_index < 4) {
            if (mr_conv->IsCurrentParamALong()) {
              *((uint32_t*)&longValue + 1) = *(uint32_t*)((char*)sp + 0x80 + gpr_index * 4);
            } else if (mr_conv->IsCurrentParamAReference()) {
              refValue = *(uint32_t*)((char*)sp + 0x80 + gpr_index * 4);
            } else {
              otherValue = *(uint32_t*)((char*)sp + 0x80 + gpr_index * 4);
            }
            
            gpr_index++;
          } else {
            if (mr_conv->IsCurrentParamALong()) {
              *((uint32_t*)&longValue + 1) = *(uint32_t*)((char*)sp + 0xb4 + off + 4);
            } else if (mr_conv->IsCurrentParamAReference()) {
              refValue = *(uint32_t*)((char*)sp + 0xb4 + off);
            } else {
              otherValue = *(uint32_t*)((char*)sp + 0xb4 + off);
            }
          }

          if (mr_conv->IsCurrentParamALong()) {
              char bufx[100] = {0};
              sprintf(bufx, "arg = %02d off = %d size = %d longValue = %x-%x", args_count, off, size, *((uint32_t*)&longValue + 1), *((uint32_t*)&longValue));
              MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
            } else if (mr_conv->IsCurrentParamAReference()) {
              char bufx[100] = {0};
              sprintf(bufx, "arg = %02d off = %d size = %d refValue = %x", args_count, off, size, refValue);
              MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());

              mirror::Object* o = reinterpret_cast<mirror::Object*>(refValue);
              if (o != nullptr) {
                  if (o->IsString()) {
                      ObjPtr<mirror::String> tmp = o->AsString();
                      std::string str = tmp->ToModifiedUtf8();
                      std::string s = "String:" + str;
                      int32_t nlen = s.length();
                      if (nlen > OUT_LENGTH) {
                        nlen = OUT_LENGTH;
                      }
                      MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                  } else if (o->IsByteArray()) {
                      ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                      void *ptmp = pAry->GetRawData(sizeof(char), 0);
                      int32_t nlen = pAry->GetLength();
                      if (nlen > OUT_LENGTH) {
                        nlen = OUT_LENGTH;
                      }
                      MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                  } else if (o->IsCharArray()) {
                      ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                      void *ptmp = pAry->GetRawData(sizeof(char), 0);
                      int32_t nlen = pAry->GetLength();
                      if (nlen > OUT_LENGTH) {
                        nlen = OUT_LENGTH;
                      }
                      MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                  } else {
                      std::unordered_set<mirror::Object*> test;
                      DumpObject(o, test);
                  }
              }
            } else {
              char bufx[100] = {0};
              sprintf(bufx, "arg = %02d off = %d size = %d otherValue = %x", args_count, off, size, otherValue);
              MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
            }          
        }

        mr_conv->Next();
      }
    }
  }
}

void PrintMirrorObj1(art::mirror::Object* o) REQUIRES_SHARED(Locks::mutator_lock_) {
    if (o != nullptr) {
          ObjPtr<mirror::Class> obj_class = o->GetClass();
          if (obj_class != nullptr) {
                if (o->IsString()) {
                ObjPtr<mirror::String> tmp = o->AsString();
                if (tmp != nullptr) {
                  std::string str = tmp->ToModifiedUtf8();
                  std::string s = "String:" + str;
                  int32_t nlen = s.length();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                }
            } else if (o->IsByteArray()) {
                ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                }
            } else if (o->IsCharArray()) {
                ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                }
            } else {
                std::unordered_set<mirror::Object*> test;
                DumpObject(o, test);
            }
          }
      }
}

void PrintMirrorObj2(art::mirror::Object* o) REQUIRES_SHARED(Locks::mutator_lock_) {
    if (o != nullptr) {
          ObjPtr<mirror::Class> obj_class = o->GetClass();
          if (obj_class != nullptr) {
                if (o->IsString()) {
                ObjPtr<mirror::String> tmp = o->AsString();
                if (tmp != nullptr) {
                  std::string str = tmp->ToModifiedUtf8();
                  std::string s = "String:" + str;
                  int32_t nlen = s.length();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                }
            } else if (o->IsByteArray()) {
                ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                }
            } else if (o->IsCharArray()) {
                ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                }
            } else {
                std::unordered_set<mirror::Object*> test;
                DumpObject(o, test);
            }
          }
      }
}

void PrintMirrorObj3(art::mirror::Object* o) REQUIRES_SHARED(Locks::mutator_lock_) {
    if (o != nullptr) {
          ObjPtr<mirror::Class> obj_class = o->GetClass();
          if (obj_class != nullptr) {
                if (o->IsString()) {
                ObjPtr<mirror::String> tmp = o->AsString();
                if (tmp != nullptr) {
                  std::string str = tmp->ToModifiedUtf8();
                  std::string s = "String:" + str;
                  int32_t nlen = s.length();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                }
            } else if (o->IsByteArray()) {
                ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                }
            } else if (o->IsCharArray()) {
                ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                }
            } else {
                std::unordered_set<mirror::Object*> test;
                DumpObject(o, test);
            }
          }
      }
}

void PrintMirrorObj4(art::mirror::Object* o) REQUIRES_SHARED(Locks::mutator_lock_) {
    if (o != nullptr) {
          ObjPtr<mirror::Class> obj_class = o->GetClass();
          if (obj_class != nullptr) {
                if (o->IsString()) {
                ObjPtr<mirror::String> tmp = o->AsString();
                if (tmp != nullptr) {
                  std::string str = tmp->ToModifiedUtf8();
                  std::string s = "String:" + str;
                  int32_t nlen = s.length();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                }
            } else if (o->IsByteArray()) {
                ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                }
            } else if (o->IsCharArray()) {
                ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                }
            } else {
                std::unordered_set<mirror::Object*> test;
                DumpObject(o, test);
            }
          }
      }
}

void PrintMirrorObj5(art::mirror::Object* o) REQUIRES_SHARED(Locks::mutator_lock_) {
    if (o != nullptr) {
          ObjPtr<mirror::Class> obj_class = o->GetClass();
          if (obj_class != nullptr) {
                if (o->IsString()) {
                ObjPtr<mirror::String> tmp = o->AsString();
                if (tmp != nullptr) {
                  std::string str = tmp->ToModifiedUtf8();
                  std::string s = "String:" + str;
                  int32_t nlen = s.length();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)s.c_str(), nlen, "[**S]:\n", gettid());
                }
            } else if (o->IsByteArray()) {
                ObjPtr<mirror::ByteArray> pAry = o->AsByteArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[B]]:\n", gettid());
                }
            } else if (o->IsCharArray()) {
                ObjPtr<mirror::CharArray> pAry = o->AsCharArray();
                if (pAry != nullptr) {
                  void *ptmp = pAry->GetRawData(sizeof(char), 0);
                  int32_t nlen = pAry->GetLength();
                  if (nlen > OUT_LENGTH) {
                    nlen = OUT_LENGTH;
                  }
                  MyWrite((unsigned char*)ptmp, nlen, "[*[C]]:\n", gettid());
                }
            } else {
                std::unordered_set<mirror::Object*> test;
                DumpObject(o, test);
            }
          }
      }
}

void artMethodEntered_ARM64(ArtMethod *method,
						  Thread *self,
						  void *sp) {
  if (!bTrace)
  {
    std::string pkg = Runtime::Current()->GetProcessPackageName();
    if (pkg == strPackageName)
    {
      if (initialize_methodnames()) {
        bTrace = true;    
      }
    }
  }

  if (bTrace)
  {
    if (method->GetIsMonitorInitialized() == nullptr)
    {
      if (check_methodname(method->PrettyMethod(true)))
      {
        method->SetIsMonitorEnabled((const void *)1);
      }
      method->SetIsMonitorInitialized((const void *)1);
    }

    ArtMethod* caller = reinterpret_cast<ArtMethod*>(*(uint64_t*)((char*)sp + 0x1F0)); //x0-x30 d0-d30 offset
    /*
    ArtMethod* caller1 = NULL;
    if (caller == NULL) {
      for (int i = 0x10; i < 0x100; i += 0x10) {
        uint64_t x30 = *(uint64_t*)((char*)sp + 0x1F0 + i + 0x28);
        if ((x30 & 0x7EC) ==  0x7EC || (x30 & 0x808) == 0x808) {
          //DumpHex((unsigned char*)((char*)sp + 0x1F0), 0x60, gettid());
          ArtMethod* result1 = (ArtMethod*)*(uint64_t*)((char*)sp + 0x1F0 + i);
          if (((uint64_t)result1 - (uint64_t)sp) < 0x10000 && ((uint64_t)result1 - (uint64_t)sp) > 0) {
            caller1 = (ArtMethod*)*(uint64_t*)(result1);
          }
          
          break;
        }
      }
    }
    */

    if (method->GetIsMonitorEnabled() != nullptr || (caller != NULL && caller->GetIsMonitorEnabled() != nullptr) /* || \
      (caller1 != NULL && caller1->GetIsMonitorEnabled() != nullptr) */)
    {
      std::string output;
      if (caller) {
        output += caller->PrettyMethod(true) + " ==> ";
        output += method->PrettyMethod(true);
        MyWrite((unsigned char*)output.c_str(), output.size(), "[CC]:", gettid());
      }
      else {
        output += method->PrettyMethod(true);
        MyWrite((unsigned char*)output.c_str(), output.size(), "[CCN]:", gettid());
        return;

        /*
        const ManagedStack* current_fragment = self->GetManagedStack();
        if (current_fragment != NULL) {
          if (current_fragment->GetTopQuickFrameGenericJniTag()) { //sp  | 1
            std::string output1;
            std::ostringstream oss;
            self->Dump(oss, false, false);
            output1 += oss.str() + "\n";
            MyWrite((unsigned char*)output1.c_str(), output1.size(), "[CCNSS]:", gettid());
          }
          else {
            MyWrite((unsigned char*)output.c_str(), output.size(), "[CCNx]:", gettid());
          }
        }
        */
      }

      /*
      if (caller1) 
      {
        output += caller1->PrettyMethod(true) + " ==> ";
        output += method->PrettyMethod(true);
        MyWrite((unsigned char*)output.c_str(), output.size(), "[IC]:", gettid());
      } 
      */

      if (g_enable_stack.size()) {
          if (self->HasManagedStack()) {
            std::string output1;
            std::ostringstream oss;
            self->Dump(oss, false, false);
            output1 += oss.str() + "\n";
            MyWrite((unsigned char*)output1.c_str(), output1.size(), "[SS]:", gettid());
          }
      }

      /*
      char buf[100] = {0};
      sprintf(buf, "%p %p %p", method, self, method->GetEntryPointFromQuickCompiledCode());
      DumpHex((unsigned char*)buf, strlen(buf), gettid());
      DumpHex((unsigned char*)sp, 0x400, gettid());
      */

      if (g_enable_args.size() == 0) {
        return;
      }


      bool is_static = method->IsStatic();
      bool is_synchronized = method->IsSynchronized();
      const char* shorty = method->GetShorty();

      MallocArenaPool pool;
      ArenaAllocator allocator(&pool);

      std::unique_ptr<ManagedRuntimeCallingConvention> mr_conv(
        ManagedRuntimeCallingConvention::Create(&allocator, is_static, is_synchronized, shorty, (art::InstructionSet)2));

      int gp_reg_index = 1;   // we start from X1/W1, X0 holds ArtMethod*.
      int fp_reg_index = 0;   // D0/S0.
      uint32_t args_count = 0;

      // We need to choose the correct register (D/S or X/W) since the managed
      // stack uses 32bit stack slots.
      mr_conv->ResetIterator(FrameOffset(0));
      while (mr_conv->HasNext()) {

        args_count++;

        FrameOffset offset = mr_conv->CurrentParamStackOffset();
        int32_t off = offset.Int32Value();
        uint32_t size = mr_conv->CurrentParamSize();

        char bufx[512] = {0};
        if (mr_conv->IsCurrentParamAFloatOrDouble()) {  // FP regs.
            double doubleValue;
            float floatValue;
            if (fp_reg_index < 8) {
              if (!mr_conv->IsCurrentParamADouble()) {
                *((float*)&floatValue) = *(float*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d floatValue = %f", args_count, off, size, floatValue);
              } else {
                *((double*)&doubleValue) = *(double*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, doubleValue);
              }
              fp_reg_index++;
            } else {  // just increase the stack offset.
              if (!mr_conv->IsCurrentParamADouble()) {
                *((float*)&floatValue) = *(float*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d floatValue = %f", args_count, off, size, floatValue);
              } else {
                *((double*)&doubleValue) = *(double*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, doubleValue);
              }
            }
            MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
        } else {  // GP regs.
          uint32_t refValue;
          uint64_t longValue;
          if (gp_reg_index < 8) {
            if (mr_conv->IsCurrentParamALong() && (!mr_conv->IsCurrentParamAReference())) {
              *((uint64_t*)&longValue) = *(uint64_t*)((char*)sp + gp_reg_index*8);
              sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, (double)longValue);
              MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
            } else {
              *((uint32_t*)&refValue) = *(uint32_t*)((char*)sp + gp_reg_index*8);
              sprintf(bufx, "arg = %02d off = %d size = %d refValue = %08x", args_count, off, size, refValue);
              MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
              if (mr_conv->IsCurrentParamAReference()) {
                mirror::Object* o = reinterpret_cast<mirror::Object*>(refValue);
                PrintMirrorObj1(o);
              }
            }
            gp_reg_index++;
          } else {  // just increase the stack offset.
            if (mr_conv->IsCurrentParamALong() && (!mr_conv->IsCurrentParamAReference())) {
                *((uint64_t*)&longValue) = *(uint64_t*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, (double)longValue);
                MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
            } else {
                *((uint32_t*)&refValue) = *(uint32_t*)((char*)sp + 0x1F0 + off);
                sprintf(bufx, "arg = %02d off = %d size = %d refValue = %08x", args_count, off, size, refValue);
                MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                if (mr_conv->IsCurrentParamAReference()) {
                  mirror::Object* o = reinterpret_cast<mirror::Object*>(refValue);
                  PrintMirrorObj2(o);
                  }
                }
            }
          }

        mr_conv->Next();
      }
    }
  }
}


void TestJniArg(ArtMethod *method, [[maybe_unused]] Thread *self, [[maybe_unused]] void *sp) REQUIRES_SHARED(Locks::mutator_lock_) {

  bool is_static = method->IsStatic();
  bool is_synchronized = method->IsSynchronized();
  //bool is_fast_native = method->IsFastNative();
  //bool is_critical_native = method->IsCriticalNative();
  const char* shorty = method->GetShorty();


  MallocArenaPool pool;
  ArenaAllocator allocator(&pool);

  //std::unique_ptr<JniCallingConvention> mr_jni_conv(
  //JniCallingConvention::Create(&allocator, is_static, is_synchronized, is_fast_native, is_critical_native, shorty, (art::InstructionSet)2));

  std::unique_ptr<ManagedRuntimeCallingConvention> mr_conv(
  ManagedRuntimeCallingConvention::Create(&allocator, is_static, is_synchronized, shorty, (art::InstructionSet)2));


  uint32_t args_count = 0;
  mr_conv->ResetIterator(FrameOffset(0));
  while (mr_conv->HasNext()) {

    args_count++;

    FrameOffset offset = mr_conv->CurrentParamStackOffset();
    int32_t off = offset.Int32Value();
    uint32_t size = mr_conv->CurrentParamSize();

    char bufx[512] = {0};
    if (mr_conv->IsCurrentParamAFloatOrDouble()) {  // FP regs.
        double doubleValue;
        float floatValue;
        // just increase the stack offset.
        if (!mr_conv->IsCurrentParamADouble()) {
          *((float*)&floatValue) = *(float*)((char*)sp + 0xE0 + off);
          sprintf(bufx, "arg = %02d off = %d size = %d floatValue = %f", args_count, off, size, floatValue);
        } else {
          *((double*)&doubleValue) = *(double*)((char*)sp + 0xE0 + off);
          sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, doubleValue);
        }
        MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
    } else {  // just increase the stack offset.
      uint32_t refValue;
      uint64_t longValue;
      if (mr_conv->IsCurrentParamALong() && (!mr_conv->IsCurrentParamAReference())) {
          *((uint64_t*)&longValue) = *(uint64_t*)((char*)sp + 0xE0 + off);
          sprintf(bufx, "arg = %02d off = %d size = %d doubleValue = %lf", args_count, off, size, (double)longValue);
          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
      } else {
          *((uint32_t*)&refValue) = *(uint32_t*)((char*)sp + 0xE0 + off);
          sprintf(bufx, "arg = %02d off = %d size = %d refValue = %08x", args_count, off, size, refValue);
          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
          if (mr_conv->IsCurrentParamAReference()) {
            mirror::Object* o = reinterpret_cast<mirror::Object*>(refValue);
            PrintMirrorObj3(o);
          }
      }
    }

    mr_conv->Next();
  }
}

bool IsNeedTrace(ArtMethod *method) REQUIRES_SHARED(Locks::mutator_lock_) {
  if (!bTrace)
  {
    std::string pkg = Runtime::Current()->GetProcessPackageName();
    if (pkg == strPackageName)
    {
      if (initialize_methodnames()) {
        bTrace = true;    
      }
    }
  }

  if (bTrace)
  {
    if (method->GetIsMonitorInitialized() == nullptr)
    {
      if (check_methodname(method->PrettyMethod(true)))
      {
        method->SetIsMonitorEnabled((const void *)1);
      }
      method->SetIsMonitorInitialized((const void *)1);
    }

    if (method->GetIsMonitorEnabled() != nullptr)
    {
      return true;
    } 
  }

  return false;
}

bool artMethodEntered_INTERPRETER(Thread *self, ShadowFrame &shadow_frame) REQUIRES_SHARED(Locks::mutator_lock_) {
  if (!bTrace)
  {
    std::string pkg = Runtime::Current()->GetProcessPackageName();
    if (pkg == strPackageName)
    {
      if (initialize_methodnames()) {
        bTrace = true;    
      }
    }
  }

  if (bTrace)
  {
    ArtMethod *method = shadow_frame.GetMethod();
    if (method->GetIsMonitorInitialized() == nullptr)
    {
      if (check_methodname(method->PrettyMethod(true)))
      {
        method->SetIsMonitorEnabled((const void *)1);
      }
      method->SetIsMonitorInitialized((const void *)1);
    }

    ShadowFrame* linker = shadow_frame.GetLink();
    ArtMethod* caller = NULL;
    if (linker) {
        caller = linker->GetMethod();
    }

    if (method->GetIsMonitorEnabled() != nullptr || (caller != NULL && caller->GetIsMonitorEnabled() != nullptr))
    {
      std::string output;

      if (linker) {
        output += caller->PrettyMethod(true) + " ==> ";
      } else {
        if (!((reinterpret_cast<long>(self) & 1) == 1)) {
          std::ostringstream oss;
          self->Dump(oss, false, false);
          output += oss.str() + "\n";
        }
      }

      output += method->PrettyMethod(true);

      if ((reinterpret_cast<long>(self) & 1) == 1) {
        MyWrite((unsigned char*)output.c_str(), output.size(), "[IC]:", gettid());
        return true;
      } else {
        MyWrite((unsigned char*)output.c_str(), output.size(), "[I]:", gettid());
        //MyWrite((unsigned char *)&g_enable_recursion_num, sizeof(size_t), "[enable_recursion]", gettid());
      }
      
      if (g_enable_stack.size()) {
          std::string output1;
          std::ostringstream oss;
          self->Dump(oss, false, false);
          output1 += oss.str() + "\n";
          MyWrite((unsigned char*)output1.c_str(), output1.size(), "[ISS]:", gettid());
      }

      if (g_enable_args.size() == 0) {
        return true;
      }

      CodeItemDataAccessor accessor(method->DexInstructionData());
      uint16_t arg_offset = accessor.RegistersSize() - accessor.InsSize();
      uint32_t* args = shadow_frame.GetVRegArgs(arg_offset);
      const char *shorty = method->GetShorty();
      if (args != nullptr && shorty[1] != '\0') {
          size_t shorty_index = 1;
          size_t arg_index = method->IsStatic() ? 0 : 1;
              while (shorty[shorty_index] != '\0') {
                  switch (shorty[shorty_index]) {
                      case 'L': {
                          mirror::Object* o = reinterpret_cast<StackReference<mirror::Object>*>(&args[arg_index])->AsMirrorPtr();
                          if (o != nullptr) {
                              PrintMirrorObj4(o);
                          }

                          /*
                          if (strstr(method->PrettyMethod(true).c_str(), "void com.pairip.licensecheck.LicenseResponseHelper.validateResponse(android.os.Bundle, java.lang.String)") != NULL) {
                            return false;
                          }

                          if (strstr(method->PrettyMethod(true).c_str(), "com.pairip.licensecheck.RepeatedCheckMetadata com.pairip.licensecheck.LicenseResponseHelper.getRepeatedCheckMetadata(android.os.Bundle)") != NULL) {
                            return false;
                          }
                          */
                          
                          

                          break;
                      }
                      case 'J':
                      case 'D': {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  J = %x-%x %lf", shorty_index, args[arg_index], args[arg_index+1], *(double*)&args);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          /*
                          if (strstr(output.c_str(), "==> boolean X.UgJ.LJ(double)") != NULL) {
                            double x = 2.0;
                            args[arg_index] = *((uint32_t*)&x);
                            args[arg_index+1] = *((uint32_t*)&x + 1);
                          }
                          */
                          arg_index++;
                          break;
                      }
                      case 'C':
                      {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  C = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          break;
                      }
                      case 'I':
                      {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  I = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          /*
                          if (strstr(output.c_str(), "==> void com.pairip.licensecheck.LicenseClient.processResponse(int, android.os.Bundle)") != NULL) {
                            int x = 0;
                            args[arg_index] = *((uint32_t*)&x); 
                          }
                          */
                          break;
                      }
                      case 'F': {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  F = %x %f", shorty_index, args[arg_index], *(float*)&args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          break;
                      }
                      default:
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  Other = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid()); 
                          break;
                      }

                      shorty_index++;
                      arg_index++;
                  } 
          }
    }
  }

  return true;
}

void TestArg(ArtMethod* m, uint32_t* args) REQUIRES_SHARED(Locks::mutator_lock_) {
      const char *shorty = m->GetShorty();
      if (args != nullptr && shorty[1] != '\0') {
          size_t shorty_index = 1;
          size_t arg_index = m->IsStatic() ? 0 : 1;
              while (shorty[shorty_index] != '\0') {
                  switch (shorty[shorty_index]) {
                      case 'L': {
                          mirror::Object* o = reinterpret_cast<StackReference<mirror::Object>*>(&args[arg_index])->AsMirrorPtr();
                          if (o != nullptr) {
                              PrintMirrorObj5(o);
                          }

                          break;
                      }
                      case 'J':
                      case 'D': {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  J = %x-%x %lf", shorty_index, args[arg_index], args[arg_index+1], *(double*)&args);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          arg_index++;
                          break;
                      }
                      case 'C':
                      {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  C = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          break;
                      }
                      case 'I':
                      {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  I = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          break;
                      }
                      case 'F': {
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  F = %x %f", shorty_index, args[arg_index], *(float*)&args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid());
                          break;
                      }
                      default:
                          char bufx[100] = {0};
                          sprintf(bufx, "arg = %02zu  Other = %x", shorty_index, args[arg_index]);
                          MyWrite((unsigned char*)bufx, strlen(bufx), "[arg]:", gettid()); 
                          break;
                      }

                      shorty_index++;
                      arg_index++;
                  } 
    }
}

void artMethodEntered_JNI(ArtMethod *method, Thread *self, ArtMethod *caller, void *sp) REQUIRES_SHARED(Locks::mutator_lock_)
{
  CHECK(method != nullptr && self != nullptr);

  if (bTrace)
  {
    std::string output;
    if (caller) {
      output += caller->PrettyMethod(true) + " ==> ";
    } 
    output += method->PrettyMethod(true);
    MyWrite((unsigned char*)output.c_str(), output.size(), "[N]:", gettid());

    if (method->GetIsMonitorInitialized() == nullptr)
    {
      if (check_methodname(method->PrettyMethod(true)))
      {
        method->SetIsMonitorEnabled((const void *)1);
      }
      method->SetIsMonitorInitialized((const void *)1);
    }

    if (caller != NULL && (caller->GetIsMonitorInitialized() == nullptr))
    {
      if (check_methodname(caller->PrettyMethod(true)))
      {
        caller->SetIsMonitorEnabled((const void *)1);
      }
      caller->SetIsMonitorInitialized((const void *)1);
    }
    
    if (method->GetIsMonitorEnabled() != nullptr || (caller != NULL && caller->GetIsMonitorEnabled() != nullptr)) {
      TestJniArg(method, self, sp);
    }
  }
}

}  // namespace art
