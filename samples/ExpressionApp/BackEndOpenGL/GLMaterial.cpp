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

#include "GLMaterial.h"
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>


#ifdef _MSC_VER
  #define strcasecmp _stricmp
#endif // _MSC_VER


 ////////////////////////////////////////////////////////////////////////////////
 /////                          UTILITY FUNCTIONS                           /////
 ////////////////////////////////////////////////////////////////////////////////


static void SplitString(const std::string &s, std::vector<std::string>&tokens) {
  tokens.clear();
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, ' ')) {
    if (0 == token.size())
      continue;
    tokens.push_back(token);
  }
}


static void SetColorFromStringArray(GLSpectrum3f& color, std::string* strs) {
  color.r = strtof(strs[0].c_str(), nullptr);
  color.g = strtof(strs[1].c_str(), nullptr);
  color.b = strtof(strs[2].c_str(), nullptr);
}


static bool StrToBool(const char* str) {
  return  strcasecmp(str, "true") == 0 ||
          strcasecmp(str, "on")   == 0 ||
          strcasecmp(str, "yes")  == 0 ||
          strcasecmp(str, "1")    == 0;
}


////////////////////////////////////////////////////////////////////////////////
/////                               GLMaterial                             /////
////////////////////////////////////////////////////////////////////////////////


GLMaterial::~GLMaterial() {
  //if (diffuseTexture)
  //    delete diffuseTexture;
}


GLMaterial::GLMaterial() {
  clear();
}


GLMaterial::GLMaterial(const GLMaterial& mtl) {
  *this = mtl;
}


void GLMaterial::setTextureFile(const char* file) {
  if (file) diffuseTextureFile = file;
  else      diffuseTextureFile.clear();
}


void GLMaterial::clear() {
  diffuseColor.set(1.f, 1.f, 1.f);
  ambientColor.set(0.f, 0.f, 0.f);
  specularColor.set(0.f, 0.f, 0.f);
  transmissionColor.set(0.f, 0.f, 0.f);
  specularExponent = 0.f;
  opacity = 1.f;
  diffuseTexture = nullptr;
  diffuseTextureFile.clear();
  illuminationModel = kUnspecifiedIlluminationModel;
}


////////////////////////////////////////////////////////////////////////////////
/////                           GLMaterialLibrary                          /////
////////////////////////////////////////////////////////////////////////////////


struct GLMaterialName {
  GLMaterial mtl;
  std::string name;
  GLMaterialName() {}
  GLMaterialName(const GLMaterial& matParam, const char* nameParam) {
    mtl = matParam;
    name = nameParam;
  }
};


struct GLMaterialLibrary::Impl {
  std::vector<GLMaterialName> lib;
};


GLMaterialLibrary::GLMaterialLibrary() {
  pimpl = new Impl;
}


GLMaterialLibrary::~GLMaterialLibrary() {
  delete pimpl;
}


void GLMaterialLibrary::clear() {
  pimpl->lib.clear();
}


unsigned GLMaterialLibrary::numMaterials() const {
  return unsigned(pimpl->lib.size());
}


NvCV_Status GLMaterialLibrary::addMaterial(const GLMaterial& mtrl, const char* name) {
  if (getMaterial(name))
    return NVCV_ERR_SELECTOR;
  pimpl->lib.emplace_back(mtrl, name);
  return NVCV_SUCCESS;
}


NvCV_Status GLMaterialLibrary::addDiffuseMaterial(const GLSpectrum3f& color, const char* name) {
  GLMaterial mtrl;
  if (getMaterial(name))
    return NVCV_ERR_SELECTOR;
  mtrl.diffuseColor = color;
  pimpl->lib.emplace_back(mtrl, name);
  return NVCV_SUCCESS;
}


NvCV_Status GLMaterialLibrary::removeMaterial(const char* name) {
  unsigned i, n;
  for (i = 0, n = unsigned(pimpl->lib.size()); i < n; ++i) {
    if (name == pimpl->lib[i].name) {
      pimpl->lib.erase(pimpl->lib.begin() + i);
      return NVCV_SUCCESS;
    }
  }
  return NVCV_ERR_FEATURENOTFOUND;
}


GLMaterial* GLMaterialLibrary::newMaterial(const char* name) {
  if (getMaterial(name))
    return nullptr;
  size_t z = pimpl->lib.size();
  pimpl->lib.resize(z + 1);
  GLMaterialName* mtn = &pimpl->lib[z];
  mtn->name = name;
  return &mtn->mtl;
}


const GLMaterial* GLMaterialLibrary::getMaterial(const char* name) const {
  GLMaterialName *mp, *mEnd;

  for (mEnd = (mp = pimpl->lib.data()) + pimpl->lib.size(); mp < mEnd; ++mp)
    if (name == mp->name)
      return &mp->mtl;
  return nullptr;
}


const GLMaterial* GLMaterialLibrary::getMaterial(unsigned i, const char** name) const {
  if (i < pimpl->lib.size()) {
    const GLMaterialName& matn = pimpl->lib[i];
    if (name)
      *name = matn.name.c_str();
    return &matn.mtl;
  }
  if (name)
    name = nullptr;
  return nullptr;
}


NvCV_Status GLMaterialLibrary::read(const char* name) {
  unsigned                  lineNum;
  std::vector<std::string>  tokens;
  GLMaterial                *mtl  = nullptr;

  clear();
  std::ifstream fd(name);
  if (!fd.is_open())
    return NVCV_ERR_READ;
  std::string line;

  for (lineNum = 1; std::getline(fd, line); ++lineNum) {
    SplitString(line, tokens);
    if (!tokens.size())
      continue;
    if (tokens[0][0] == '#') {
      continue;
    }
    if (tokens[0] == "newmtl" && 2 == tokens.size()) {
      mtl = newMaterial(tokens[1].c_str());
      continue;
    }
    if (tokens[0] == "Ka" && 4 == tokens.size()) {
      SetColorFromStringArray(mtl->ambientColor, &tokens[1]);
      continue;
    }
    if (tokens[0] == "Kd" && 4 == tokens.size()) {
      SetColorFromStringArray(mtl->diffuseColor, &tokens[1]);
      continue;
    }
    if (tokens[0] == "Ks" && 4 == tokens.size()) {
      SetColorFromStringArray(mtl->specularColor, &tokens[1]);
      continue;
    }
    if (tokens[0] == "Tf" && 4 == tokens.size()) {
      SetColorFromStringArray(mtl->transmissionColor, &tokens[1]);
      continue;
    }
    if (tokens[0] == "illum" && 2 == tokens.size()) {
      mtl->illuminationModel = (unsigned char)strtol(tokens[1].c_str(), nullptr, 10);
      continue;
    }
    if (tokens[0] == "d" && 2 == tokens.size()) {
      /* We don't support "_d" */
      mtl->opacity = strtof(tokens[1].c_str(), nullptr);
      continue;
    }
    if (tokens[0] == "Ns" && 2 == tokens.size()) {   /* We don't support "-d" */
      mtl->specularExponent = strtof(tokens[1].c_str(), nullptr);
      continue;
    }
    if (tokens[0] == "sharpness") {   /* We don't support sharpness */
      continue;
    }
    if (tokens[0] == "Ni") {   /* We don't support index of refraction */
      continue;
    }
    if (tokens[0] == "map_Kd") {
      for (unsigned i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "-blendu") {
          ++i;
        }
        else if (tokens[i] == "-blendv") {
          ++i;
        }
        else if (tokens[i] == "-cc") {
          ++i;
        }
        else if (tokens[i] == "-clamp") {
          ++i;
        }
        else if (tokens[i] == "-mm") {
          ++i;
        }
        else if (tokens[i] == "-o") {
          i += 3;
        }
        else if (tokens[i] == "-s") {
          i += 3;
        }
        else if (tokens[i] == "-t") {
          i += 3;
        }
        else if (tokens[i] == "-texres") {
          ++i;
        }
        else if (tokens[i][0] == '-') {
          printf("Unknown option: \"%s\"\n", tokens[i].c_str());
        }
        else {
          mtl->diffuseTextureFile = tokens[i];
        }
      }
      if (mtl->diffuseTextureFile.empty())
        printf("No diffuse texture given on line %u\n", lineNum);
      continue;
    }
  }

  return NVCV_SUCCESS;
}