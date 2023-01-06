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

#ifndef __MESH_RENDERER__
#define __MESH_RENDERER__

#include "nvCVImage.h"
#include <string>
#include <vector>


/// Abstract class to provide methods and hide the implementation.
class MeshRenderer {
public:
  struct Dispatch {
    // We get these procs from the DLL
    NvCV_Status (*name)(const char **str);
    NvCV_Status (*info)(const char **str);
    NvCV_Status (*create)(MeshRenderer **han);
    void        (*destroy)(MeshRenderer *han);
    NvCV_Status (*read)(MeshRenderer *han, const char *modelFile);
    NvCV_Status (*init)(MeshRenderer *han, unsigned width, unsigned height, const char *windowName);
    NvCV_Status (*setCamera)(MeshRenderer *han, const float locPt[3], const float lookVec[3], const float upVec[3],
                             float vfov, float near_z, float far_z);
    NvCV_Status (*render)(MeshRenderer *han,
                          const float exprs[53], const float qrot[4], const float trans[3], NvCVImage *result);
    Dispatch();
    ~Dispatch() {}
  };

  /// Destructor.
  void destroy();

  /// Get the name of the renderer associated with the given handle.
  /// @param[out] str a place to store a pointer to the name of the mesh renderer. 
  /// @return     NVCV_SUCCESS  if the specified mesh renderer was successfully instantiated.
  NvCV_Status name(const char **str) const;

  /// Get information about the renderer.
  /// @param[out] str a place to store a pointer to the name of the mesh renderer. 
  /// @return     NVCV_SUCCESS  if the specified mesh renderer was successfully instantiated.
  NvCV_Status info(const char **str) const;

  /// Read the specified mesh model.
  /// If another model was already loaded, it will first be unloaded.
  /// If a relative path is specified, several places are searched.
  /// @param[in]  modelFile the name of the file containing the desired mesh model.
  /// @return     NVCV_SUCCESS  if the model was read successfully.
  NvCV_Status read(const char *modelFile);

  /// Initialize the rendering resources.
  /// @param[in] width      The desired width  of the rendered image.
  /// @param[in] height     The desired height of the rendered image.
  /// @param[in] windowName The name given to the auxiliary window, if the renderer requires one.
  /// @return               NVCV_SUCCESS  if the renderer was successfully initialized.
  NvCV_Status init(unsigned width, unsigned height, const char *windowName);

  /// Set the viewing parameters.
  /// @param[in]  locPt   the 3D point location of the camera.
  ///                     NULL implies the default location, derived from the rest pose of the model.
  /// @param[in]  lookVec the 3D vector indicating the direction of view. This does not need to be normalized.
  ///                     NULL implies the default direction, derived from the rest pose of the model.
  /// @param[in]  upVec   the 3D vector pointing up. This does not need to be normalized.
  ///                     NULL implies the default up direction, derived from the rest pose of the model.
  /// @param[in]  vfov    the vertical field of view of the camera. Zero implies an orthographic camera.
  /// @param[in]  near_z  Near z clipping plane. Distance along camera's negative z-axis. If both near_z and far_z are
  /// 0.0, the z clipping planes are computed based on the mesh bounding box.
  /// @param[in]  far_z   Far z clipping plane. Distance along camera's negative z-axis. If both near_z and far_z are
  /// 0.0, the z clipping planes are computed based on the mesh bounding box.
  /// @return     NVCV_SUCCESS  if the camera was initialized successfully.
  NvCV_Status setCamera(const float locPt[3], const float lookVec[3], const float upVec[3], float vfov, float near_z = 0.0f, float far_z = 0.0f);

  /// Render the mesh as deformed by the expression signals.
  /// @param[in]  exprs   the expression signals (53 of them).
  /// @param[in]  qrot    the rotation    of the model as an xyzw quaternion.
  /// @param[in]  trans   the translation of the model as an xyz  vector.
  /// @param[out] result  the resultant rendered image.
  /// @note: This will appear upside-down.
  NvCV_Status render(const float exprs[53], const float qrot[4], const float trans[3], NvCVImage *result);

protected:
  MeshRenderer()  {}   ///< Never create a member of this class
  ~MeshRenderer() {}   ///< Instantiations are always subclasses

  Dispatch m_dispatch;
};

class MeshRendererBroker {
public:

  /// Constructor.
  MeshRendererBroker();

  /// Destructor.
  ~MeshRendererBroker();

  /// Set the directory to be searched for additional renderers.
  /// @param[in] dir  the directory to be searched for additional renderers.
  /// @return         +NVCV_SUCCESS if the operation was successful.
  NvCV_Status setRendererDirectory(const char *dir);

  /// Return a list of the available mesh renderers.
  /// @param[out] list      pointer to a place to store the list of available renderers, separated by newlines.
  /// @return NVCV_SUCCESS  if a list was successfully returned.
  NvCV_Status getMeshRendererList(std::vector<std::string>& list);

  /// Retrieve the information about the selected renderer.
  /// @param[in]  renderer the selected renderer.
  /// @param[out] info     a place to store a pointer to the information about the selected renderer.
  /// @return     NVCV_SUCCESS              if the information was successfully retrieved;
  ///             NVCV_ERR_FEATURENOTFOUND  if the specified renderer was not found.
  /// @note       This string is ephemeral. If persistency is desired, a copy must be made. The previous pointer is
  ///             invalidated when this is called repeatedly
  NvCV_Status info(const char *renderer, const char **info);

  /// Create an instance of the chosen mesh renderer.
  /// @param[in]  renderer the desired renderer. NULL chooses the default renderer.
  /// @param[out] han      a place to store a handle to the desired mesh renderer.
  /// @return     NVCV_SUCCESS              if the specified mesh renderer was successfully instantiated.
  ///             NVCV_ERR_FEATURENOTFOUND  if the specified renderer was not found.
  NvCV_Status create(const char *renderer, MeshRenderer **han);

  /// Add a new renderer to the broker's portfolio.
  /// @param disp the renderer's dispatch table.
  void addRenderer(MeshRenderer::Dispatch *disp);

private:
  class Impl;
  Impl *m_impl;
};

#endif // __MESH_RENDERER__

