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

  TCHAR path[MAX_PATH], fullPath[2*MAX_PATH];

  // There can be multiple apps on the system,
  // some might include the SDK in the app package and
  // others might expect the SDK to be installed in Program Files
  GetEnvironmentVariable(TEXT("NV_AR_SDK_PATH"), path, MAX_PATH);
  if (_tcscmp(path, TEXT("USE_APP_PATH"))) {
    // App has not set environment variable to "USE_APP_PATH"
    // So pick up the SDK dll and dependencies from Program Files
    GetEnvironmentVariable(TEXT("ProgramFiles"), path, MAX_PATH);
    size_t max_len = sizeof(fullPath)/sizeof(TCHAR);
    _stprintf_s(fullPath, max_len, TEXT("%s\\NVIDIA Corporation\\NVIDIA AR SDK\\"), path);
    SetDllDirectory(fullPath);
  }
  static const HINSTANCE NvArLib = nvLoadLibrary("nvARPose");
  return NvArLib;
}

NvCV_Status NvAR_API NvCVImage_Init(NvCVImage* im, unsigned width, unsigned height, int pitch, void* pixels,
                                       NvCVImage_PixelFormat format, NvCVImage_ComponentType type, unsigned isPlanar,
                                       unsigned onGPU) {
  static const auto funcPtr = (decltype(NvCVImage_Init)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Init");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, pitch, pixels, format, type, isPlanar, onGPU);
}

void NvAR_API NvCVImage_InitView(NvCVImage* subImg, NvCVImage* fullImg, int x, int y, unsigned width,
                                   unsigned height) {
  static const auto funcPtr = (decltype(NvCVImage_InitView)*)nvGetProcAddress(getNvARLib(), "NvCVImage_InitView");

  if (nullptr != funcPtr) funcPtr(subImg, fullImg, x, y, width, height);
}

NvCV_Status NvAR_API NvCVImage_Alloc(NvCVImage* im, unsigned width, unsigned height, NvCVImage_PixelFormat format,
                              NvCVImage_ComponentType type, unsigned isPlanar, unsigned onGPU, unsigned alignment) {
  static const auto funcPtr = (decltype(NvCVImage_Alloc)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Alloc");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, format, type, isPlanar, onGPU, alignment);
}

NvCV_Status NvAR_API NvCVImage_Realloc(NvCVImage* im, unsigned width, unsigned height,
                                          NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
                                          unsigned isPlanar, unsigned onGPU, unsigned alignment) {
  static const auto funcPtr = (decltype(NvCVImage_Realloc)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Realloc");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(im, width, height, format, type, isPlanar, onGPU, alignment);
}

void NvAR_API NvCVImage_Dealloc(NvCVImage* im) {
  static const auto funcPtr = (decltype(NvCVImage_Dealloc)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Dealloc");

  if (nullptr != funcPtr) funcPtr(im);
}

NvCV_Status NvAR_API NvCVImage_Create(unsigned width, unsigned height, NvCVImage_PixelFormat format,
                                         NvCVImage_ComponentType type, unsigned isPlanar, unsigned onGPU,
                                         unsigned alignment, NvCVImage** out) {
  static const auto funcPtr = (decltype(NvCVImage_Create)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Create");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(width, height, format, type, isPlanar, onGPU, alignment, out);
}

void NvAR_API NvCVImage_Destroy(NvCVImage* im) {
  static const auto funcPtr = (decltype(NvCVImage_Destroy)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Destroy");

  if (nullptr != funcPtr) funcPtr(im);
}

void NvAR_API NvCVImage_ComponentOffsets(NvCVImage_PixelFormat format, int* rOff, int* gOff, int* bOff, int* aOff,
                                           int* yOff) {
  static const auto funcPtr =
      (decltype(NvCVImage_ComponentOffsets)*)nvGetProcAddress(getNvARLib(), "NvCVImage_ComponentOffsets");

  if (nullptr != funcPtr) funcPtr(format, rOff, gOff, bOff, aOff, yOff);
}

NvCV_Status NvAR_API NvCVImage_Transfer(const NvCVImage* src, NvCVImage* dst, float scale, CUstream_st* stream,
                                           NvCVImage* tmp) {
  static const auto funcPtr = (decltype(NvCVImage_Transfer)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Transfer");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, dst, scale, stream, tmp);
}

NvCV_Status NvAR_API NvCVImage_Composite(const NvCVImage* src, const NvCVImage* mat, NvCVImage* dst) {
  static const auto funcPtr = (decltype(NvCVImage_Composite)*)nvGetProcAddress(getNvARLib(), "NvCVImage_Composite");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, mat, dst);
}

NvCV_Status NvAR_API NvCVImage_CompositeOverConstant(const NvCVImage* src, const NvCVImage* mat,
                                                        const unsigned char bgColor[3], NvCVImage* dst) {
  static const auto funcPtr =
      (decltype(NvCVImage_CompositeOverConstant)*)nvGetProcAddress(getNvARLib(), "NvCVImage_CompositeOverConstant");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, mat, bgColor, dst);
}

NvCV_Status NvAR_API NvCVImage_FlipY(const NvCVImage* src, NvCVImage* dst) {
  static const auto funcPtr = (decltype(NvCVImage_FlipY)*)nvGetProcAddress(getNvARLib(), "NvCVImage_FlipY");

  if (nullptr == funcPtr) return NVCV_ERR_LIBRARY;
  return funcPtr(src, dst);
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
  static const auto funcPtr = (decltype(NvAR_GetF32Array)*)nvGetProcAddress(getNvARLib(), "NvAR_GetCNvAR_GetF32ArrayudaStream");

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

#ifdef _WIN32
__declspec(dllexport) const char* __cdecl
#else
const char*
#endif  // _WIN32 or linux
    NvCV_GetErrorStringFromCode(NvCV_Status code) {
  static const auto funcPtr =
      (decltype(NvCV_GetErrorStringFromCode)*)nvGetProcAddress(getNvARLib(), "NvCV_GetErrorStringFromCode");

  if (nullptr == funcPtr) return "Cannot find nvARPose DLL or its dependencies";
  return funcPtr(code);
}
