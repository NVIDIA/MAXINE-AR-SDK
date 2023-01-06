/*###############################################################################
#
# Copyright 2021 NVIDIA Corporation
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


#ifdef _WIN32
  #define _WINSOCKAPI_
  #include <windows.h>
  #include <tchar.h>
#else // UNIX
  #include <dlfcn.h>
  typedef void *HMODULE;
  typedef void *HANDLE;
  typedef void *HINSTANCE;
#endif // _WIN32 || UNIX

#include "MeshRenderer.h"
#include "DirectoryIterator.h"
#include "OpenGLMeshRenderer.h" // Eventually this will be discoverable
#include <string.h>
#include <string>
#include <vector>


#ifdef _WIN32
  #define nvLoadLibrary(library) LoadLibrary(TEXT(library))
#else // UNIX
  #define nvLoadLibrary(library) dlopen(library, RTLD_LAZY)
#endif // _WIN32 || UNIX


inline void* nvGetProcAddress(HINSTANCE handle, const char *proc) {
  if (nullptr == handle) return nullptr;
  #ifdef _WIN32
    return GetProcAddress(handle, proc);
  #else // UNIX
    return dlsym(handle, proc);
  #endif // _WIN32 || UNIX
}


inline int nvFreeLibrary(HINSTANCE handle) {
  if (nullptr == handle) return -1;
  #ifdef _WIN32
    return int(!FreeLibrary(handle));     // convert bool true to 0 int and 1 error code
  #else // UNIX
    return dlclose(handle);               // 0 on success, error code otherwise
  #endif // _WIN32 || UNIX
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                         Abstract class MeshRenderer                  /////
/////        This merely converts from a C++ to a C object call.           /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG_CAST // We never instantiate MeshRenderer, only ClubMeshRenderer, so we don't need a dynamic cast.
  #define static_cast dynamic_cast  // But #define this to prove it
#endif // DEBUG_CAST


NvCV_Status MeshRenderer::name(const char **str) const {
  return m_dispatch.name(str);
}


NvCV_Status MeshRenderer::info(const char **str) const {
  return m_dispatch.info(str);
}


void MeshRenderer::destroy() {
  m_dispatch.destroy(this);
}


NvCV_Status MeshRenderer::read(const char *modelFile) {
  return m_dispatch.read(this, modelFile);
}


NvCV_Status MeshRenderer::init(unsigned width, unsigned height, const char *windowName) {
  return m_dispatch.init(this, width, height, windowName);
}


NvCV_Status MeshRenderer::setCamera(const float locPt[3], const float lookVec[3], const float upVec[3], float vfov, float z_near, float z_far) {
  return m_dispatch.setCamera(this, locPt, lookVec, upVec, vfov, z_near, z_far);
}


NvCV_Status MeshRenderer::render(const float exprs[53], const float qrot[4], const float trans[3], NvCVImage *result) {
  return m_dispatch.render(this, exprs, qrot, trans, result);
}


MeshRenderer::Dispatch::Dispatch() {
  memset(this, 0, sizeof(*this));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                             MeshRendererBroker                       /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class RendererInfo {
public:
  std::string name, info, file;
  HMODULE module;
  MeshRenderer::Dispatch dispatch;
  RendererInfo() {
    memset(&dispatch, 0, sizeof(dispatch));
    module = nullptr;
  }
};


class MeshRendererBroker::Impl {
public:
  static const char nameStr[], infoStr[], createStr[], destroyStr[], readStr[], initStr[], setCameraStr[], renderStr[];

  std::string rendererDirectory;
  std::vector<RendererInfo> renderers;

  NvCV_Status GetRenderers();
  NvCV_Status LoadRenderer(const char *name);
  NvCV_Status UnloadRenderer(const char *name);
  bool AlreadyHave(const char *name);
  ~Impl() {
    for (RendererInfo& ri : renderers)
      (void)UnloadRenderer(ri.name.c_str());
  }
};
const char MeshRendererBroker::Impl::nameStr[]      = "RendererName";
const char MeshRendererBroker::Impl::infoStr[]      = "RendererInfo";
const char MeshRendererBroker::Impl::createStr[]    = "RendererName";
const char MeshRendererBroker::Impl::destroyStr[]   = "RendererDestroy";
const char MeshRendererBroker::Impl::readStr[]      = "RendererReadModel";
const char MeshRendererBroker::Impl::initStr[]      = "RendererInit";
const char MeshRendererBroker::Impl::setCameraStr[] = "RendererSetCamera";
const char MeshRendererBroker::Impl::renderStr[]    = "RendererRender";


static bool HasDLLSuffix(const char *name) {
  static const char dllSuffix[] =
    #ifdef _WIN32
      ".dll"
    #else // UNIX
      ".so"
    #endif // _WIN32 || UNIX
  ;
  constexpr unsigned suf_len = sizeof(dllSuffix) - 1; // Strings have a NULL terminator, are not part of the string
  unsigned name_len = unsigned(strlen(name));
  return name_len > suf_len && !strcmp(name + name_len - suf_len, dllSuffix);
}


bool MeshRendererBroker::Impl::AlreadyHave(const char* name) {
  for (const RendererInfo& ri : renderers)
    if (ri.name == name)
      return true;
  return false;
}


NvCV_Status MeshRendererBroker::Impl::GetRenderers() {
  if (rendererDirectory.empty())    // Or should we pass in the directory as an argument?
    return NVCV_ERR_INITIALIZATION; // The renderer directory was not initialized.
  DirectoryIterator dit(rendererDirectory.c_str(), DirectoryIterator::kTypeFile);
  const char *fileName;
  unsigned type;
  while (0 == dit.next(&fileName, &type)) {
    if (!HasDLLSuffix(fileName))
      continue;
    std::string path = rendererDirectory + '/' + fileName;
    HINSTANCE lib = nvLoadLibrary(path.c_str());
    if (!lib)
      continue;
    typedef NvCV_Status(*GetStringFunc)(const char** str);
    GetStringFunc getString;
    const char *name = nullptr, *info = nullptr;
    if ((nullptr != (getString = reinterpret_cast<GetStringFunc>(nvGetProcAddress(lib, nameStr)))) &&
        (NVCV_SUCCESS == (*getString)(&name))                                                      &&
        (nullptr != (getString = reinterpret_cast<GetStringFunc>(nvGetProcAddress(lib, infoStr)))) &&
        (NVCV_SUCCESS == (*getString)(&info))                                                      &&
        !AlreadyHave(name)
    ) {
      unsigned i = unsigned(renderers.size());
      renderers.resize(i + 1);
      RendererInfo& ri = renderers[i];
      ri.name.assign(name);
      ri.info.assign(info);
      ri.file.assign(path);
    }
    nvFreeLibrary(lib);
  }
  return NVCV_SUCCESS;
}


NvCV_Status MeshRendererBroker::Impl::LoadRenderer(const char *name) {
  for (RendererInfo& ri : renderers) {
    if (name == ri.name) {
      if (!ri.module) {
        ri.module = nvLoadLibrary(ri.file.c_str());
        // BE VERY CAREFUL WHEN CHANGING FUNCTION SIGNATURES, ESPECIALLY FOR WINDOWS!!!! THERE ARE NO CHECKS BELOW!!!!
        *((void**)&ri.dispatch.name)      = nvGetProcAddress(ri.module, nameStr);
        *((void**)&ri.dispatch.info)      = nvGetProcAddress(ri.module, infoStr);
        *((void**)&ri.dispatch.create)    = nvGetProcAddress(ri.module, createStr);
        *((void**)&ri.dispatch.destroy)   = nvGetProcAddress(ri.module, destroyStr);
        *((void**)&ri.dispatch.read)      = nvGetProcAddress(ri.module, readStr);
        *((void**)&ri.dispatch.init)      = nvGetProcAddress(ri.module, initStr);
        *((void**)&ri.dispatch.setCamera) = nvGetProcAddress(ri.module, setCameraStr);
        *((void**)&ri.dispatch.render)    = nvGetProcAddress(ri.module, renderStr);
      }
      return NVCV_SUCCESS;
    }
  }
  return NVCV_ERR_FEATURENOTFOUND;
}


NvCV_Status MeshRendererBroker::Impl::UnloadRenderer(const char *name) {
  NvCV_Status err = NVCV_ERR_FEATURENOTFOUND;
  for (RendererInfo& ri : renderers) {
    if (name == ri.name) {
      err = !ri.module ? NVCV_SUCCESS : nvFreeLibrary(ri.module) ? NVCV_ERR_LIBRARY : NVCV_SUCCESS;
      ri.module = nullptr;
      memset(&ri.dispatch, 0, sizeof(ri.dispatch));
      break;
    }
  }
  return err;
}


MeshRendererBroker::MeshRendererBroker() {
  NvCV_Status err;
  m_impl = new Impl;

  // Automatically register the OpenGL Renderer
  MeshRenderer::Dispatch disp;
  if (NVCV_SUCCESS == (err = OpenGLMeshRenderer_InitDispatch(&disp)))
    MeshRendererBroker::addRenderer(&disp);
}


MeshRendererBroker::~MeshRendererBroker() {
  delete m_impl;
}


NvCV_Status MeshRendererBroker::setRendererDirectory(const char *dir) {
  // Load more renderers from DLLs in the given directory
  m_impl->rendererDirectory = dir;
  return m_impl->GetRenderers();
}


NvCV_Status MeshRendererBroker::getMeshRendererList(std::vector<std::string>& list) {
  list.clear();
  list.resize(m_impl->renderers.size());
  for (size_t i = 0; i < list.size(); ++i)
    list[i] = m_impl->renderers[i].name;
  return list.size() ? NVCV_SUCCESS : NVCV_ERR_FEATURENOTFOUND;
}


NvCV_Status MeshRendererBroker::info(const char *renderer, const char **info) {
  if (!info)
    return NVCV_ERR_PARAMETER;
  for (const RendererInfo& ri : m_impl->renderers) {
    if (ri.name == renderer) {
      *info = ri.info.c_str();
      return NVCV_SUCCESS;
    }
  }
  *info = nullptr;
  return NVCV_ERR_FEATURENOTFOUND;
}


NvCV_Status MeshRendererBroker::create(const char *renderer, MeshRenderer **han) {
  if (!han)
    return NVCV_ERR_PARAMETER;
  for (const RendererInfo& ri : m_impl->renderers) {
    if (ri.name == renderer) {
      if (!ri.dispatch.create) {
        NvCV_Status err = m_impl->LoadRenderer(renderer);
        if (NVCV_SUCCESS != err)
          return err;
      }
      return ri.dispatch.create(han);
    }
  }
  *han = nullptr;
  return NVCV_ERR_FEATURENOTFOUND;
}


void MeshRendererBroker::addRenderer(MeshRenderer::Dispatch *disp) {
  unsigned n = unsigned(m_impl->renderers.size());
  m_impl->renderers.resize(n + 1);
  RendererInfo& ri = m_impl->renderers[n];
  ri.dispatch = *disp;
  const char *str;
  disp->name(&str); ri.name = str;
  disp->info(&str); ri.info = str;
}
