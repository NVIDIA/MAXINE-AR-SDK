#if defined(linux) || defined(unix) || defined(__linux)
#warning nvCVImageProxy.cpp not ported
#else
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
#include "../include/vpi/Status.h"
#include "../include/vpi/VPI.h"
#include "../include/vpi/CUDAInterop.h"
#include "../include/vpi/experimental/ColorNames.h"
#include "../include/vpi/experimental/HOG.h"

#ifdef _WIN32
#define _WINSOCKAPI_
#include <windows.h>
#include <tchar.h>
#else // !_WIN32
#include <dlfcn.h>
typedef void* HMODULE;
typedef void* HANDLE;
typedef void* HINSTANCE;
#endif // _WIN32

// Parameter string does not include the file extension
#ifdef _WIN32
#define nvLoadLibrary(library) LoadLibrary(TEXT(library ".dll"))
#else // !_WIN32
#define nvLoadLibrary(library) dlopen("lib" library ".so", RTLD_LAZY)
#endif // _WIN32


inline void* nvGetProcAddress(HINSTANCE handle, const char* proc) {
    if (nullptr == handle) return nullptr;
#ifdef _WIN32
    return GetProcAddress(handle, proc);
#else // !_WIN32
    return dlsym(handle, proc);
#endif // _WIN32
}

inline int nvFreeLibrary(HINSTANCE handle) {
#ifdef _WIN32
    return FreeLibrary(handle);
#else
    return dlclose(handle);
#endif
}

HINSTANCE getVPILib() {
    TCHAR path[MAX_PATH], tmpPath[MAX_PATH], fullPath[MAX_PATH];
    static HINSTANCE VPILib = NULL;
    static bool bSDKPathSet = false;
    if (!bSDKPathSet) {
        VPILib = nvLoadLibrary("nvvpi2");
        if (VPILib)  bSDKPathSet = true;
    }
    if (!bSDKPathSet) {
        // There can be multiple apps on the system,
        // some might include the SDK in the app package and
        // others might expect the SDK to be installed in Program Files
        GetEnvironmentVariable(TEXT("NV_VIDEO_EFFECTS_PATH"), path, MAX_PATH);
        GetEnvironmentVariable(TEXT("NV_AR_SDK_PATH"), tmpPath, MAX_PATH);
        if (_tcscmp(path, TEXT("USE_APP_PATH")) && _tcscmp(tmpPath, TEXT("USE_APP_PATH"))) {
            // App has not set environment variable to "USE_APP_PATH"
            // So pick up the SDK dll and dependencies from Program Files
            GetEnvironmentVariable(TEXT("ProgramFiles"), path, MAX_PATH);
            size_t max_len = sizeof(fullPath) / sizeof(TCHAR);
            _stprintf_s(fullPath, max_len, TEXT("%s\\NVIDIA Corporation\\NVIDIA Video Effects\\"), path);
            SetDllDirectory(fullPath);
            VPILib = nvLoadLibrary("nvvpi2");
            if (!VPILib) {
                _stprintf_s(fullPath, max_len, TEXT("%s\\NVIDIA Corporation\\NVIDIA AR SDK\\"), path);
                SetDllDirectory(fullPath);
                VPILib = nvLoadLibrary("nvvpi2");
            }
        }
        bSDKPathSet = true;
    }
    return VPILib;
}

const char *vpiStatusGetName(VPIStatus code) {
  static const auto funcPtr = (decltype(vpiStatusGetName) *)nvGetProcAddress(getVPILib(), "vpiStatusGetName");

  if (nullptr == funcPtr) return nullptr;
  return funcPtr(code);
}

VPIStatus vpiGetLastStatusMessage(char *msgBuffer, int32_t lenBuffer)
{
  static const auto funcPtr = (decltype(vpiGetLastStatusMessage) *)nvGetProcAddress(getVPILib(), "vpiGetLastStatusMessage");

  if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
  return funcPtr(msgBuffer, lenBuffer);
}

VPIStatus vpiStreamCreate(uint32_t flags, VPIStream *stream) {
    static const auto funcPtr = (decltype(vpiStreamCreate)*)nvGetProcAddress(getVPILib(), "vpiStreamCreate");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(flags, stream);
}

void vpiStreamDestroy(VPIStream stream) {
    static const auto funcPtr = (decltype(vpiStreamDestroy)*)nvGetProcAddress(getVPILib(), "vpiStreamDestroy");

    if (nullptr == funcPtr) return;
    return funcPtr(stream);
}

VPIStatus vpiStreamSync(VPIStream stream) {
    static const auto funcPtr = (decltype(vpiStreamSync)*)nvGetProcAddress(getVPILib(), "vpiStreamSync");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(stream);
}

void vpiPayloadDestroy(VPIPayload payload) {
    static const auto funcPtr = (decltype(vpiPayloadDestroy)*)nvGetProcAddress(getVPILib(), "vpiPayloadDestroy");

    if (nullptr == funcPtr) return;
    return funcPtr(payload);
}

void vpiImageDestroy(VPIImage img) {
    static const auto funcPtr = (decltype(vpiImageDestroy)*)nvGetProcAddress(getVPILib(), "vpiImageDestroy");

    if (nullptr == funcPtr) return;
    return funcPtr(img);
}

VPIStatus vpiCreateExtractColorNameFeatures(uint32_t backends, VPIImageFormat outType, VPIPayload *payload) {
    static const auto funcPtr = (decltype(vpiCreateExtractColorNameFeatures)*)nvGetProcAddress(getVPILib(), "vpiCreateExtractColorNameFeatures");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(backends, outType, payload);
}

VPIStatus vpiSubmitExtractColorNameFeatures(VPIStream stream, uint32_t backend, VPIPayload payload,
    VPIImage input, VPIImage *output, int32_t numOutputs) {
    static const auto funcPtr = (decltype(vpiSubmitExtractColorNameFeatures)*)nvGetProcAddress(getVPILib(), "vpiSubmitExtractColorNameFeatures");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(stream, backend, payload, input, output, numOutputs);
}

VPIStatus vpiCreateExtractHOGFeatures(uint32_t backends, int32_t width, int32_t height, int32_t features,
    int32_t cellSize, int32_t numOrientations, int32_t *outNumFeatures,
    VPIPayload *payload) {
    static const auto funcPtr = (decltype(vpiCreateExtractHOGFeatures)*)nvGetProcAddress(getVPILib(), "vpiCreateExtractHOGFeatures");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(backends, width, height, features, cellSize, numOrientations, outNumFeatures, payload);
}

VPIStatus vpiCreateExtractHOGFeaturesBatch(uint32_t backends, int32_t maxBatchWidth, int32_t maxBatchHeight,
  int32_t imgWidth, int32_t imgHeight, int32_t features, int32_t cellSize,
  int32_t numOrientations, int32_t *outNumFeatures, VPIPayload *payload)
{
  static const auto funcPtr =
      (decltype(vpiCreateExtractHOGFeaturesBatch) *)nvGetProcAddress(getVPILib(), "vpiCreateExtractHOGFeaturesBatch");

  if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
  return funcPtr(backends, maxBatchWidth, maxBatchHeight, imgWidth, imgHeight, features, cellSize, numOrientations, outNumFeatures, payload);
}

VPIStatus vpiSubmitExtractHOGFeatures(VPIStream stream, uint32_t backend, VPIPayload payload, VPIImage input,
    VPIImage *outFeatures, int32_t numFeatures) {
    static const auto funcPtr = (decltype(vpiSubmitExtractHOGFeatures)*)nvGetProcAddress(getVPILib(), "vpiSubmitExtractHOGFeatures");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(stream, backend, payload, input, outFeatures, numFeatures);
}

VPIStatus vpiImageCreateCUDAMemWrapper(const VPIImageData *cudaData, uint32_t flags, VPIImage *img) {
    static const auto funcPtr = (decltype(vpiImageCreateCUDAMemWrapper)*)nvGetProcAddress(getVPILib(), "vpiImageCreateCUDAMemWrapper");

    if (nullptr == funcPtr) return VPI_ERROR_NOT_IMPLEMENTED;
    return funcPtr(cudaData, flags, img);
}


#endif // enabling for this file
