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
#ifndef __SIMPLE_FACE_MODEL__
#define __SIMPLE_FACE_MODEL__

#include <stdint.h>

#include <vector>
#include <string>

#include "FaceIO.h"
#include "nvAR_defs.h"

/********************************************************************************
 * SimpleFaceModel
 ********************************************************************************/


struct SimpleFaceModel {
  std::vector<NvAR_Point3f>    shapeMean;
  std::vector<NvAR_Vector3f>   shapeModes;  /* shapeMean.size() * numModes */
  std::vector<float>           shapeEigenValues;
  std::vector<NvAR_Vector3u16> triangles;
  struct BlendShape {
    std::string                name;
    std::vector<NvAR_Vector3f> shape;
  };
  std::vector<BlendShape>      blendShapes;
  struct Partition {
    unsigned    partitionIndex;    ///< The index of the partition.
    unsigned    faceIndex;         ///< The index of the first face in the partition.
    unsigned    numFaces;          ///< The number of faces in the partition.
    unsigned    vertexIndex;       ///< The index of the first topological vertex in the partition.
    unsigned    numVertexIndices;  ///< The number of topological vertices in the partition.
    int         smoothingGroup;    ///< Smoothing group > 0; no smoothing == 0; undefined < 0.
    std::string name;              ///< The name of the partition.
    std::string materialName;      ///< The name of the material assigned to the partition.
    void set(unsigned partIx, unsigned firstFaceIndex, unsigned lastFaceIndex,
      unsigned firstVertexIndex, unsigned lastVertexIndex, int smooth,
      const char* partName = nullptr, const char* mtrlName = nullptr) {
      partitionIndex = partIx;
      smoothingGroup = smooth;
      faceIndex = firstFaceIndex;
      numFaces = lastFaceIndex - firstFaceIndex + 1;
      vertexIndex = firstVertexIndex;
      numVertexIndices = lastVertexIndex - firstVertexIndex + 1;
      if (partName) name = partName;
      if (mtrlName) materialName = mtrlName;
    }
    Partition(unsigned partIx, unsigned firstFaceIndex, unsigned lastFaceIndex,
      unsigned firstVertexIndex, unsigned lastVertexIndex, int smooth,
      const char* partName, const char* mtrlName) {
      set(partIx, firstFaceIndex, lastFaceIndex, firstVertexIndex, lastVertexIndex, smooth, partName, mtrlName);
    }
    Partition() { set(0, 0, 0, 0, 0, -1, nullptr, nullptr); }
  };
  std::vector<Partition>       partitions;
  std::vector<unsigned short>  ibugLandmarkMappings; /* 68 */
  const unsigned short         ibugRightContour[8] = { 1,  2,  3,  4,  5,  6,  7,  8 };
  const unsigned short         ibugLeftContour[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };
  std::vector<unsigned short>  modelRightContour;
  std::vector<unsigned short>  modelLeftContour;
  std::vector<unsigned short>  adjacentFaces;
  std::vector<unsigned short>  adjacentVertices;
  std::vector<unsigned short>  nvlmLandmarks;
  std::vector<unsigned short>  nvlmRightContour;
  std::vector<unsigned short>  nvlmLeftContour;

  void appendMode(const NvAR_Point3f* pts) {
    size_t  n = shapeMean.size(),
      off = shapeModes.size();
    shapeModes.resize(off + n);
    float       *to = shapeModes[off].vec;      // Delta mode vector
    const float *fr = &pts->x;                  // Mode points
    const float *mn = &shapeMean[0].x;          // Mean points
    for (n *= 3; n--;)                          // 3D points
      *to++ = *fr++ - *mn++;                    // Delta shape
  }

  void setBlendShape(unsigned i, const std::string& name, const NvAR_Point3f* pts) {
    size_t n = shapeMean.size();
    blendShapes[i].name = name;
    blendShapes[i].shape.resize(n);
    float       *to = blendShapes[i].shape.data()->vec; // Delta mode vector
    const float *fr = &pts->x;                          // Blendshape points
    const float *mn = &shapeMean[0].x;                  // Mean points
    for (n *= 3; n--;)                                  // 3D points
      *to++ = *fr++ - *mn++;                            // Delta shape
  }
};


/********************************************************************************
 * SimpleFaceModelAdapter
 ********************************************************************************/

class SimpleFaceModelAdapter : public FaceIOAdapter {
public:
  SimpleFaceModel fm;

  uint32_t  getShapeMeanSize() const override { return unsigned(fm.shapeMean.size()) * 3; }
  uint32_t  getShapeModesSize() const override { return unsigned(fm.shapeModes.size()) * 3; }
  uint32_t  getShapeNumModes() const override { return unsigned(fm.shapeModes.size() / fm.shapeMean.size()); }
  uint32_t  getShapeEigenvaluesSize()const override { return unsigned(fm.shapeEigenValues.size()); }
  float*    getShapeMean(uint32_t size) override { if (size) fm.shapeMean.resize(size / 3);
                return &fm.shapeMean.data()->x; };
  float*    getShapeModes(uint32_t modeSize, uint32_t numModes) override {
                if (modeSize) fm.shapeModes.resize(modeSize / 3 * numModes); return fm.shapeModes.data()->vec; }
  float*    getShapeEigenvalues(uint32_t numModes) override { if (numModes) fm.shapeEigenValues.resize(numModes);
                return fm.shapeEigenValues.data(); }

  uint32_t  getColorMeanSize() const override { return 0; }
  uint32_t  getColorModesSize() const override { return 0; }
  uint32_t  getColorNumModes() const override { return 0; }
  uint32_t  getColorEigenvaluesSize() const override { return 0; }
  float*    getColorMean(uint32_t /*size*/) override { return nullptr; }
  float*    getColorModes(uint32_t /*modeSize*/, uint32_t /*numModes*/) override { return nullptr; }
  float*    getColorEigenvalues(uint32_t /*numModes*/) override { return nullptr; }

  void      setTriangleListSize(uint32_t size) override { fm.triangles.resize(size / 3); }
  uint32_t  getTriangleListSize() const override { return unsigned(fm.triangles.size()) * 3; }
  uint16_t* getTriangleList(uint32_t size) override { if (size) fm.triangles.resize(size / 3);
                return fm.triangles.data()->vec; }

  void      setTextureCoordinatesSize(uint32_t /*size*/) override {}
  uint32_t  getTextureCoordinatesSize() const override { return 0; }
  float*    getTextureCoordinates(uint32_t /*size*/) override { return nullptr; }

  void      setNumBlendShapes(uint32_t n) override { fm.blendShapes.resize(n); }
  void      setBlendShapeName(uint32_t i, const char* name) override { fm.blendShapes[i].name = name; }
  uint32_t  getNumBlendShapes() const override { return unsigned(fm.blendShapes.size()); }
  const char* getBlendShapeName(uint32_t i) const override { return fm.blendShapes[i].name.c_str(); }
  uint32_t  getBlendShapeSize(uint32_t i) const override { return unsigned((fm.blendShapes[i].shape.size()) * 3); }
  float*    getBlendShape(uint32_t i, uint32_t size) override { if (size) fm.blendShapes[i].shape.resize(size / 3);
                return fm.blendShapes[i].shape.data()->vec; }

  void      setIbugLandmarkMappingsSize(uint32_t n) override { fm.ibugLandmarkMappings.resize(n); }
  uint32_t  getIbugLandmarkMappingsSize() const override { return unsigned(fm.ibugLandmarkMappings.size()); }
  uint16_t* getIbugLandmarkMappings(uint32_t size) override { if (size) fm.ibugLandmarkMappings.resize(size);
                return fm.ibugLandmarkMappings.data(); }
  void      appendIbugLandmarkMapping(uint16_t i) override { fm.ibugLandmarkMappings.push_back(i); }
  void      appendIbugLandmarkMapping(uint16_t i, uint16_t j) override { fm.ibugLandmarkMappings.push_back(i);
                fm.ibugLandmarkMappings.push_back(j); }

  void      setIbugRightContourSize(uint32_t /*n*/) override {}
  uint32_t  getIbugRightContourSize() const override {
                return sizeof(fm.ibugRightContour) / sizeof(fm.ibugRightContour[0]); }
  uint16_t* getIbugRightContour(uint32_t /*size*/) override { return const_cast<uint16_t*>(fm.ibugRightContour); }
  void      appendIbugRightContour(uint16_t /*i*/) override {}

  void      setIbugLeftContourSize(uint32_t /*n*/) override {}
  uint32_t  getIbugLeftContourSize() const override { return sizeof(fm.ibugLeftContour)/sizeof(fm.ibugLeftContour[0]);}
  uint16_t* getIbugLeftContour(uint32_t /*size*/) override { return const_cast<uint16_t*>(fm.ibugLeftContour); }
  void      appendIbugLeftContour(uint16_t /*i*/) override {}

  void      setModelRightContourSize(uint32_t n) override { fm.modelRightContour.resize(n); }
  uint32_t  getModelRightContourSize() const override { return unsigned(fm.modelRightContour.size()); }
  uint16_t* getModelRightContour(uint32_t size) override { if (size) fm.modelRightContour.resize(size);
                return fm.modelRightContour.data(); }
  void      appendModelRightContour(uint16_t i) override { fm.modelRightContour.push_back(i); }

  void      setModelLeftContourSize(uint32_t n) override { fm.modelLeftContour.resize(n); }
  uint32_t  getModelLeftContourSize() const override { return unsigned(fm.modelLeftContour.size()); }
  uint16_t* getModelLeftContour(uint32_t size) override { if (size) fm.modelLeftContour.resize(size);
                return fm.modelLeftContour.data(); }
  void      appendModelLeftContour(uint16_t i) override { fm.modelLeftContour.push_back(i); }

  void      setAdjacentFacesSize(uint32_t n) override { fm.adjacentFaces.resize(n); }
  uint32_t  getAdjacentFacesSize() const override { return unsigned(fm.adjacentFaces.size()); }
  uint16_t* getAdjacentFaces(uint32_t size) override { if (size) fm.adjacentFaces.resize(size);
                return fm.adjacentFaces.data(); }
  void      appendAdjacentFace(uint16_t i) override { fm.adjacentFaces.push_back(i); }
  void      appendAdjacentFaces(uint16_t i, uint16_t j) override { fm.adjacentFaces.push_back(i);
                fm.adjacentFaces.push_back(j); }

  void      setAdjacentVerticesSize(uint32_t n) override { fm.adjacentVertices.resize(n); }
  uint32_t  getAdjacentVerticesSize() const override { return unsigned(fm.adjacentVertices.size()); }
  uint16_t* getAdjacentVertices(uint32_t size) override { if (size) fm.adjacentVertices.resize(size);
                return fm.adjacentVertices.data(); }
  void      appendAdjacentVertex(uint16_t i) override { fm.adjacentVertices.push_back(i); }
  void      appendAdjacentVertices(uint16_t i, uint16_t j) override { fm.adjacentVertices.push_back(i);
                fm.adjacentVertices.push_back(j); }

  void      setNvlmLandmarksSize(uint32_t n) override { fm.nvlmLandmarks.resize(n); }
  uint32_t  getNvlmLandmarksSize() const override { return (uint32_t)fm.nvlmLandmarks.size(); }
  uint16_t* getNvlmLandmarks(uint32_t size) override { if (size) fm.nvlmLandmarks.resize(size);
                return fm.nvlmLandmarks.data(); }
  void      appendNvlmLandmark(uint16_t i) override { fm.nvlmLandmarks.push_back(i); }

  void      setNvlmRightContourSize(uint32_t n) override { fm.nvlmRightContour.resize(n); }
  uint32_t  getNvlmRightContourSize() const override { return (uint32_t)fm.nvlmRightContour.size(); }
  uint16_t* getNvlmRightContour(uint32_t size) override { if (size) fm.nvlmRightContour.resize(size);
                return fm.nvlmRightContour.data(); }
  void      appendNvlmRightContour(uint16_t i) override { fm.nvlmRightContour.push_back(i); }

  void      setNvlmLeftContourSize(uint32_t n) override { fm.nvlmLeftContour.resize(n); }
  uint32_t  getNvlmLeftContourSize() const override { return (uint32_t)fm.nvlmLeftContour.size(); }
  uint16_t* getNvlmLeftContour(uint32_t size) override { if (size) fm.nvlmLeftContour.resize(size);
                return fm.nvlmLeftContour.data(); }
  void      appendNvlmLeftContour(uint16_t i) override { fm.nvlmLeftContour.push_back(i); }

  void      setNumPartitions(uint32_t n) override { fm.partitions.resize(n); }
  void      setPartitionName(uint32_t i, const char* name) override { fm.partitions.at(i).name = name; }
  void      setPartitionMaterialName(uint32_t i, const char* name) override { fm.partitions.at(i).materialName = name;}
  void      setPartition(uint32_t i, uint32_t faceIndex, uint32_t numFaces, uint32_t vertexIndex, uint32_t numVertices,
                int32_t smoothingGroup) override {
              fm.partitions.at(i).set(i, faceIndex, faceIndex + numFaces - 1, vertexIndex,
                  vertexIndex + numVertices - 1, smoothingGroup);
            }
  uint32_t    getNumPartitions() const override { return (uint32_t)fm.partitions.size(); }
  const char* getPartitionName(uint32_t i) const override { return fm.partitions.at(i).name.c_str(); }
  const char* getPartitionMaterialName(uint32_t i) const override { return fm.partitions.at(i).materialName.c_str(); }
  int16_t     getPartition(uint32_t i, uint32_t* faceIndex, uint32_t* numFaces, uint32_t* vertexIndex,
                   uint32_t* numVertices, int32_t* smoothingGroup) const override {
                const SimpleFaceModel::Partition& pt = fm.partitions.at(i);
                if (faceIndex) *faceIndex = pt.faceIndex;
                if (numFaces) *numFaces = pt.numFaces;
                if (vertexIndex) *vertexIndex = pt.vertexIndex;
                if (numVertices) *numVertices = pt.numVertexIndices;
                if (smoothingGroup) *smoothingGroup = pt.smoothingGroup;
                return (int16_t)pt.partitionIndex;
              }
};

#endif // __SIMPLE_FACE_MODEL__
