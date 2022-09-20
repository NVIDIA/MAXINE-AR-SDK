/*###############################################################################
#
# Copyright 2020 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
###############################################################################*/
#include <string>

#include "nvAR.h"

#ifdef _WIN32
#define _WINSOCKAPI_
#include <windows.h>
#include <tchar.h>
#else
#include <dlfcn.h>
typedef void* HMODULE;
typedef void* HANDLE;
typedef void* HINSTANCE;
#endif

// Parameter string does not include the file extension
#ifdef _WIN32
#define nvLoadLibrary(library) LoadLibrary(TEXT(library ".dll"))
#else
#define nvLoadLibrary(library) dlopen("lib" library ".so", RTLD_LAZY)
#endif


inline void* nvGetProcAddress(HINSTANCE handle, const char* proc) {
  if (nullptr == handle) return nullptr;
#ifdef _WIN32
  return GetProcAddress(handle, proc);
#else
  return dlsym(handle, proc);
#endif
}

inline int nvFreeLibrary(HINSTANCE handle) {
#ifdef _WIN32
  return FreeLibrary(handle);
#else
  return dlclose(handle);
#endif
}

HINSTANCE getNvARLib() {

  TCHAR path[MAX_PATH], fullPath[MAX_PATH];
  bool bSDKPathSet = false;

  extern char* g_nvARSDKPath;
  if (g_nvARSDKPath && g_nvARSDKPath[0]) {
#ifndef UNICODE
    strncpy_s(fullPath, MAX_PATH, g_nvARSDKPath, MAX_PATH);
#else
    size_t res = 0;
    mbstowcs_s(&res, fullPath, MAX_PATH, g_nvARSDKPath, MAX_PATH);
#endif
    SetDllDirectory(fullPath);
    bSDKPathSet = true;
  }

  if (!bSDKPathSet) {
  
    // There can be multiple apps on the system,
    // some might include the SDK in the app package and
    // others might expect the SDK to be installed in Program Files
    GetEnvironmentVariable(TEXT("NV_AR_SDK_PATH"), path, MAX_PATH);
    if (_tcscmp(path, TEXT("USE_APP_PATH"))) {
      // App has not set environment variable to "USE_APP_PATH"
      // So pick up the SDK dll and dependencies from Program Files
      GetEnvironmentVariable(TEXT("ProgramFiles"), path, MAX_PATH);
      size_t max_len = sizeof(fullPath) / sizeof(TCHAR);
      _stprintf_s(fullPath, max_len, TEXT("%s\\NVIDIA Corporation\\NVIDIA AR SDK\\"), path);
      SetDllDirectory(fullPath);
    }
  }
  static const HINSTANCE NvArLib = nvLoadLibrary("nvARPose");
  return NvArLib;
}

NvCV_Status NvAR_API NvAR_GetVersion(unsigned int* version) {
  static const auto funcPtr = (decltype(NvAR_GetVersion)*)nvGetProcAddress(getNvARLib(), "NvAR_GetVersion");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(version);
}

NvCV_Status NvAR_API NvAR_Create(NvAR_FeatureID featureID, NvAR_FeatureHandle* handle) {
  static const auto funcPtr = (decltype(NvAR_Create)*)nvGetProcAddress(getNvARLib(), "NvAR_Create");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(featureID, handle);
}

NvCV_Status NvAR_API NvAR_Destroy(NvAR_FeatureHandle handle) {
  static const auto funcPtr = (decltype(NvAR_Destroy)*)nvGetProcAddress(getNvARLib(), "NvAR_Destroy");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle);
}

NvCV_Status NvAR_API NvAR_SetU32(NvAR_FeatureHandle handle, const char* name, unsigned int val) {
  static const auto funcPtr = (decltype(NvAR_SetU32)*)nvGetProcAddress(getNvARLib(), "NvAR_SetU32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_SetS32(NvAR_FeatureHandle handle, const char* name, int val) {
  static const auto funcPtr = (decltype(NvAR_SetS32)*)nvGetProcAddress(getNvARLib(), "NvAR_SetS32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_SetF32(NvAR_FeatureHandle handle, const char* name, float val) {
  static const auto funcPtr = (decltype(NvAR_SetF32)*)nvGetProcAddress(getNvARLib(), "NvAR_SetF32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_SetF64(NvAR_FeatureHandle handle, const char* name, double val) {
  static const auto funcPtr = (decltype(NvAR_SetF64)*)nvGetProcAddress(getNvARLib(), "NvAR_SetF64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_SetU64(NvAR_FeatureHandle handle, const char* name, unsigned long long val) {
  static const auto funcPtr = (decltype(NvAR_SetU64)*)nvGetProcAddress(getNvARLib(), "NvAR_SetU64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_SetObject(NvAR_FeatureHandle handle, const char* name, void* ptr, unsigned long typeSize) {
  static const auto funcPtr = (decltype(NvAR_SetObject)*)nvGetProcAddress(getNvARLib(), "NvAR_SetObject");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, ptr, typeSize);
}

NvCV_Status NvAR_API NvAR_SetString(NvAR_FeatureHandle handle, const char* name, const char* str) {
  static const auto funcPtr = (decltype(NvAR_SetString)*)nvGetProcAddress(getNvARLib(), "NvAR_SetString");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, str);
}

NvCV_Status NvAR_API NvAR_SetCudaStream(NvAR_FeatureHandle handle, const char* name, CUstream stream) {
  static const auto funcPtr = (decltype(NvAR_SetCudaStream)*)nvGetProcAddress(getNvARLib(), "NvAR_SetCudaStream");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, stream);
}

NvCV_Status NvAR_API NvAR_SetF32Array(NvAR_FeatureHandle handle, const char* name, float* val, int count) {
  static const auto funcPtr = (decltype(NvAR_SetF32Array)*)nvGetProcAddress(getNvARLib(), "NvAR_SetF32Array");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val, count);
}

NvCV_Status NvAR_API NvAR_GetU32(NvAR_FeatureHandle handle, const char* name, unsigned int* val) {
  static const auto funcPtr = (decltype(NvAR_GetU32)*)nvGetProcAddress(getNvARLib(), "NvAR_GetU32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_GetS32(NvAR_FeatureHandle handle, const char* name, int* val) {
  static const auto funcPtr = (decltype(NvAR_GetS32)*)nvGetProcAddress(getNvARLib(), "NvAR_GetS32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_GetF32(NvAR_FeatureHandle handle, const char* name, float* val) {
  static const auto funcPtr = (decltype(NvAR_GetF32)*)nvGetProcAddress(getNvARLib(), "NvAR_GetF32");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_GetF64(NvAR_FeatureHandle handle, const char* name, double* val) {
  static const auto funcPtr = (decltype(NvAR_GetF64)*)nvGetProcAddress(getNvARLib(), "NvAR_GetF64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_GetU64(NvAR_FeatureHandle handle, const char* name, unsigned long long* val) {
  static const auto funcPtr = (decltype(NvAR_GetU64)*)nvGetProcAddress(getNvARLib(), "NvAR_GetU64");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, val);
}

NvCV_Status NvAR_API NvAR_GetObject(NvAR_FeatureHandle handle, const char* name, const void** ptr, unsigned long typeSize) {
  static const auto funcPtr = (decltype(NvAR_GetObject)*)nvGetProcAddress(getNvARLib(), "NvAR_GetObject");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, ptr, typeSize);
}

NvCV_Status NvAR_API NvAR_GetString(NvAR_FeatureHandle handle, const char* name, const char** str) {
  static const auto funcPtr = (decltype(NvAR_GetString)*)nvGetProcAddress(getNvARLib(), "NvAR_GetString");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, str);
}

NvCV_Status NvAR_API NvAR_GetCudaStream(NvAR_FeatureHandle handle, const char* name, const CUstream* stream) {
  static const auto funcPtr = (decltype(NvAR_GetCudaStream)*)nvGetProcAddress(getNvARLib(), "NvAR_GetCudaStream");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, stream);
}

NvCV_Status NvAR_API NvAR_GetF32Array(NvAR_FeatureHandle handle, const char* name, const float** vals, int* count) {
  static const auto funcPtr = (decltype(NvAR_GetF32Array)*)nvGetProcAddress(getNvARLib(), "NvAR_GetF32Array");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle, name, vals, count);
}

NvCV_Status NvAR_API NvAR_Run(NvAR_FeatureHandle handle) {
  static const auto funcPtr = (decltype(NvAR_Run)*)nvGetProcAddress(getNvARLib(), "NvAR_Run");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle);
}

NvCV_Status NvAR_API NvAR_Load(NvAR_FeatureHandle handle) {
  static const auto funcPtr = (decltype(NvAR_Load)*)nvGetProcAddress(getNvARLib(), "NvAR_Load");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(handle);
}

NvCV_Status NvAR_API NvAR_CudaStreamCreate(CUstream* stream) {
  static const auto funcPtr =
      (decltype(NvAR_CudaStreamCreate)*)nvGetProcAddress(getNvARLib(), "NvAR_CudaStreamCreate");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(stream);
}

NvCV_Status NvAR_API NvAR_CudaStreamDestroy(CUstream stream) {
  static const auto funcPtr =
      (decltype(NvAR_CudaStreamDestroy)*)nvGetProcAddress(getNvARLib(), "NvAR_CudaStreamDestroy");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(stream);
}