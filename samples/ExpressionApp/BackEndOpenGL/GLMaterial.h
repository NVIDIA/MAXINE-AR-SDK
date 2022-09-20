/*###############################################################################
#
# Copyright 2016-2021 NVIDIA Corporation
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

#ifndef __GLMATERIAL_H
#define __GLMATERIAL_H

#include <string>

#include "GLSpectrum.h"
#include "nvCVStatus.h"

////////////////////////////////////////////////////////////////////////////////
/// Specification for light transport on the surfaces of objects.
/// @todo Store the provenance of the material?
/// @todo Should we store the name, too?
////////////////////////////////////////////////////////////////////////////////

class GLMaterial {
public:
  /// Default constructor.
  GLMaterial();

  /// Copy constructor.
  /// @param[in]  mtl the material to copy.
  /// @note       a copy is made of the diffuseTextureFile, if not NULL.
  GLMaterial(const GLMaterial& mtl);

  /// Destructor.
  /// @note       the diffuseTextureFile string is disposed.
  /// @note       the opaque diffuseTexture is *not* disposed.
  ~GLMaterial();

  /// Assignment.
  /// This copies the diffuseTextureFile, if not NULL>
  /// @param[in]  mtl the material to copy (RHS).
  /// @return     a reference to the LHS of the assignment.
  //GLMaterial& operator=(const GLMaterial& mtl);  // default implementation

  /// Reset as it was in the constructor: 0 materials.
  void clear();

  /// Use this to set the diffuseTextureFile, by making a copy of the specified string.
  /// @param[in]  fileName    the file name of the texture file. A copy of the string is made.
  void setTextureFile(const char* fileName);

  GLSpectrum3f    ambientColor;           ///< The   ambient    color, in [0,1].
  GLSpectrum3f    diffuseColor;           ///< The   diffuse    color, in [0,1].
  GLSpectrum3f    specularColor;          ///< The   specular   color, in [0,1].
  GLSpectrum3f    transmissionColor;      ///< The transmission color, in [0,1].
  float           specularExponent;       ///< The specular exponent, in [1, 10000]
  float           opacity;                ///< The opacity, in [0,1].
  std::string     diffuseTextureFile;     ///< The name of the diffuse texture file. Set through setTextureFile().
  void            *diffuseTexture;        ///< User-defined texture representation -- unmanaged.
  unsigned char   illuminationModel;      ///< The illumination model, in [0,10], or kUnspecifiedIlluminationModel.
  static const int kUnspecifiedIlluminationModel = 255;   ///< The value to be used for an unspecified illumination model.
};


////////////////////////////////////////////////////////////////////////////////
/// Library of material specifications for surface light transport.
////////////////////////////////////////////////////////////////////////////////

class GLMaterialLibrary {
public:
  /// Constructor.
  GLMaterialLibrary();

  /// Destructor.
  ~GLMaterialLibrary();

  /// Reset as it was in the constructor: 0 materials.
  void clear();

  /// Read from a file
  NvCV_Status read(const char* name);

  /// Add a new material to the library. A copy is made both of material and name.
  /// @param[in]  mtrl    the material to be added to the library.
  /// @param[in]  name    the name of the material, for future access.
  /// @return     keErrNone       if the operation was completed successfully.
  /// @return     keErrDuplicate  if a material of the same name is already found in the library.
  NvCV_Status addMaterial(const GLMaterial& mtrl, const char* name);

  /// Add a new diffuse material to the library. A copy is made both of the name.
  /// @param[in]  color   the diffuse color to be added to the library.
  /// @param[in]  name    the name of the material, for future access.
  /// @return     keErrNone       if the operation was completed successfully.
  /// @return     keErrDuplicate  if a material of the same name is already found in the library.
  NvCV_Status addDiffuseMaterial(const GLSpectrum3f& color, const char* name);

  /// Remove a material.
  /// @param[in]  name    the name of the material to remove.
  /// @return     keErrNone       if the operation was completed successfully.
  NvCV_Status removeMaterial(const char* name);

  /// Create a new material with the given name.
  /// @param[in]  name    the name of the material, for future access.
  /// @return     a pointer to the new material in the database, if the operation was completed successfully.
  /// @return     NULL, if a material of the same name is already found in the library.
  GLMaterial* newMaterial(const char* name);

  /// Get the number of materials in the material library.
  /// @return the number of materials.
  unsigned numMaterials() const;

  /// Get the material with the specified name.
  /// @param[in]  name    the name of the material to get.
  /// @return     the specified material, or NULL if the material was not found.
  const GLMaterial* getMaterial(const char* name) const;

  /// Get the material with the specified index.
  /// @param[in]  i       the index of the material to get.
  /// @param[out] name    the name of the material with the specified index (can be NULL).
  /// @return     the specified material, or NULL if the material was not found.
  const GLMaterial* getMaterial(unsigned i, const char** name = nullptr) const;

private:
  struct Impl;
  Impl *pimpl;
};


#endif /* __GLMATERIAL_H */
