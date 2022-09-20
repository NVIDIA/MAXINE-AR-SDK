/*###############################################################################
#
# Copyright 2019-2021 NVIDIA Corporation
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

#ifndef __FACE_IO__
#define __FACE_IO__

#include <stdint.h>

enum FaceIOErr {
  kIOErrNone,
  kIOErrFileNotFound,
  kIOErrFileOpen,
  kIOErrEOF,
  kIOErrRead,
  kIOErrWrite,
  kIOErrSyntax,
  kIOErrFormat,
  kIOErrNotValue,
  kIOErrNullPointer,
  kIOErrParameter,
};

const char* FaceIOErrorStringFromCode(FaceIOErr err);

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                               IO Adapter                             *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

/********************************************************************************
 * FaceIOAdapter.
 * Subclass from this and supply the accessors.
 ********************************************************************************/

class FaceIOAdapter {
public:
  virtual uint32_t getShapeMeanSize() const { return 0; }  /* The size of the mean shape mean, in elements. */
  virtual uint32_t getShapeModesSize() const { return 0; } /* The total size of all shape modes (numModes*modeSize) */
  virtual uint32_t getShapeNumModes() const { return 0; }  /* The number of shape modes. */
  virtual uint32_t getShapeEigenvaluesSize() const { return 0; } /* The number of shape eigenvalues
                                                        (should equal the number of modes) */
  virtual float* getShapeMean(uint32_t /*size*/) { return nullptr; } /* Get a pointer to the shape mean.
                                                                 If a nonzero size if supplied, it is resized first. */
  virtual float* getShapeModes(uint32_t /*modeSize*/, uint32_t /*numModes*/) { return nullptr; } /* Get a pointer to the
                                                        shape modes, resizing first, if the parameters are nonzero. */
  virtual float* getShapeEigenvalues(uint32_t /*numModes*/) { return nullptr; } /* Get a pointer to the shape eigenvalues,
                                                             resizing first if numModes is nonzero. */

  virtual uint32_t getColorMeanSize() const { return 0; }  /* The color mean ... */
  virtual uint32_t getColorModesSize() const { return 0; } /* ... and modes */
  virtual uint32_t getColorNumModes() const { return 0; }
  virtual uint32_t getColorEigenvaluesSize() const { return 0; }
  virtual float* getColorMean(uint32_t /*size*/) { return nullptr; }
  virtual float* getColorModes(uint32_t /*modeSize*/, uint32_t /*numModes*/) { return nullptr; }
  virtual float* getColorEigenvalues(uint32_t /*numModes*/) { return nullptr; }

  virtual void setTriangleListSize(uint32_t /*size*/) {} /* The triangle list */
  virtual uint32_t getTriangleListSize() const = 0;
  virtual uint16_t* getTriangleList(uint32_t /*size*/) { return nullptr; }

  virtual void setTextureCoordinatesSize(uint32_t /*size*/) {} /* The texture coordinates */
  virtual uint32_t getTextureCoordinatesSize() const { return 0; }
  virtual float* getTextureCoordinates(uint32_t /*size*/) { return nullptr; }

  virtual void setNumBlendShapes(uint32_t /*numShapes*/) {} /* The blend shapes */
  virtual void setBlendShapeName(uint32_t /*i*/, const char* /*name*/) {}
  virtual uint32_t getNumBlendShapes() const { return 0; }
  virtual const char* getBlendShapeName(uint32_t /*i*/) const { return nullptr; }
  virtual uint32_t getBlendShapeSize(uint32_t /*i*/) const { return 0; }
  virtual float* getBlendShape(uint32_t /*i*/, uint32_t /*size*/) { return nullptr; }

  virtual void setIbugLandmarkMappingsSize(uint32_t /*n*/) {} /* The mappings from IBUG landmarks to vertex index */
  virtual uint32_t getIbugLandmarkMappingsSize() const { return 0; }
  virtual uint16_t* getIbugLandmarkMappings(uint32_t /*size*/) { return nullptr; }
  virtual void appendIbugLandmarkMapping(uint16_t /*i*/) {}
  virtual void appendIbugLandmarkMapping(uint16_t /*i*/, uint16_t /*j*/) {}

  virtual void setIbugRightContourSize(uint32_t /*n*/) {} /* The IBUG contour on the right side of the face */
  virtual uint32_t getIbugRightContourSize() const { return 0; }
  virtual uint16_t* getIbugRightContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendIbugRightContour(uint16_t /*i*/) {}

  virtual void setIbugLeftContourSize(uint32_t /*n*/) {} /* The IBUG contour on the left side of the face */
  virtual uint32_t getIbugLeftContourSize() const { return 0; }
  virtual uint16_t* getIbugLeftContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendIbugLeftContour(uint16_t /*i*/) {}

  virtual void setModelRightContourSize(uint32_t /*n*/) {} /* The right contour of our model */
  virtual uint32_t getModelRightContourSize() const { return 0; }
  virtual uint16_t* getModelRightContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendModelRightContour(uint16_t /*i*/) {}

  virtual void setModelLeftContourSize(uint32_t /*n*/) {} /* The left contour of our model */
  virtual uint32_t getModelLeftContourSize() const { return 0; }
  virtual uint16_t* getModelLeftContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendModelLeftContour(uint16_t /*i*/) {}

  virtual void setAdjacentFacesSize(uint32_t /*n*/) {} /* The topology of adjacent faces to each edge */
  virtual uint32_t getAdjacentFacesSize() const { return 0; }
  virtual uint16_t* getAdjacentFaces(uint32_t /*size*/) { return nullptr; }
  virtual void appendAdjacentFace(uint16_t /*i*/) {}
  virtual void appendAdjacentFaces(uint16_t /*i*/, uint16_t /*j*/) {}

  virtual void setAdjacentVerticesSize(uint32_t /*n*/) {} /* The topology of adjacent vertices to each edge */
  virtual uint32_t getAdjacentVerticesSize() const { return 0; }
  virtual uint16_t* getAdjacentVertices(uint32_t /*size*/) { return nullptr; }
  virtual void appendAdjacentVertex(uint16_t /*i*/) {}
  virtual void appendAdjacentVertices(uint16_t /*i*/, uint16_t /*j*/) {}

  virtual void setNvlmLandmarksSize(uint32_t /*n*/) {}      /* The tracked landmarks */
  virtual uint32_t getNvlmLandmarksSize() const { return 0; }
  virtual uint16_t* getNvlmLandmarks(uint32_t /*size*/) { return nullptr; }
  virtual void appendNvlmLandmark(uint16_t /*i*/) {}

  virtual void setNvlmRightContourSize(uint32_t /*n*/) {}   /* The tracked right jawline contour */
  virtual uint32_t getNvlmRightContourSize() const { return 0; }
  virtual uint16_t* getNvlmRightContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendNvlmRightContour(uint16_t /*i*/) {}

  virtual void setNvlmLeftContourSize(uint32_t /*n*/) {};    /* The tracked left jawline contour */
  virtual uint32_t getNvlmLeftContourSize() const { return 0; }
  virtual uint16_t* getNvlmLeftContour(uint32_t /*size*/) { return nullptr; }
  virtual void appendNvlmLeftContour(uint16_t /*i*/) {}

  virtual void setNumPartitions(uint32_t /*n*/) {}
  virtual void setPartitionName(uint32_t /*i*/, const char* /*name*/) {}
  virtual void setPartitionMaterialName(uint32_t /*i*/, const char* /*name*/) {}
  virtual void setPartition(uint32_t /*i*/, uint32_t /*faceIndex*/, uint32_t /*numFaces*/,
                            uint32_t /*vertexIndex*/, uint32_t /*numVertices*/, int32_t /*smoothingGroup*/) {}
  virtual uint32_t getNumPartitions() const { return 0; }
  virtual const char* getPartitionName(uint32_t /*i*/) const { return nullptr; }
  virtual const char* getPartitionMaterialName(uint32_t /*i*/) const { return nullptr; }
  virtual int16_t getPartition(uint32_t /*i*/, uint32_t* faceIndex, uint32_t* numFaces, uint32_t* vertexIndex,
                               uint32_t* numVertices, int32_t* smoothingGroup) const
  { if (faceIndex) *faceIndex = 0u; if (numFaces) *numFaces = 0u; if (vertexIndex) *vertexIndex = 0u;
    if (numVertices) *numVertices = 0u; if (smoothingGroup) *smoothingGroup = -1; return /*partitionIndex*/-1;
  }


  /* Const accessors do not have the ability to resize. */
  const float* getShapeMean() const { return const_cast<FaceIOAdapter*>(this)->getShapeMean(0); }
  const float* getShapeModes() const { return const_cast<FaceIOAdapter*>(this)->getShapeModes(0, 0); }
  const float* getShapeEigenvalues() const { return const_cast<FaceIOAdapter*>(this)->getShapeEigenvalues(0); }
  const float* getColorMean() const { return const_cast<FaceIOAdapter*>(this)->getColorMean(0); }
  const float* getColorModes() const { return const_cast<FaceIOAdapter*>(this)->getColorModes(0, 0); }
  const float* getColorEigenvalues() const { return const_cast<FaceIOAdapter*>(this)->getColorEigenvalues(0); }
  const float* getTextureCoordinates() const { return const_cast<FaceIOAdapter*>(this)->getTextureCoordinates(0); }
  const uint16_t* getTriangleList() const { return const_cast<FaceIOAdapter*>(this)->getTriangleList(0); }
  const float* getBlendShape(uint32_t i) const { return const_cast<FaceIOAdapter*>(this)->getBlendShape(i, 0); }
  const uint16_t* getIbugLandmarkMappings() const { return const_cast<FaceIOAdapter*>(this)->getIbugLandmarkMappings(0); }
  const uint16_t* getIbugRightContour() const { return const_cast<FaceIOAdapter*>(this)->getIbugRightContour(0); }
  const uint16_t* getIbugLeftContour() const { return const_cast<FaceIOAdapter*>(this)->getIbugLeftContour(0); }
  const uint16_t* getModelRightContour() const { return const_cast<FaceIOAdapter*>(this)->getModelRightContour(0); }
  const uint16_t* getModelLeftContour() const { return const_cast<FaceIOAdapter*>(this)->getModelLeftContour(0); }
  const uint16_t* getAdjacentFaces() const { return const_cast<FaceIOAdapter*>(this)->getAdjacentFaces(0); }
  const uint16_t* getAdjacentVertices() const { return const_cast<FaceIOAdapter*>(this)->getAdjacentVertices(0); }
  const uint16_t* getNvlmLandmarks() const { return const_cast<FaceIOAdapter*>(this)->getNvlmLandmarks(0); }
  const uint16_t* getNvlmRightContour() const { return const_cast<FaceIOAdapter*>(this)->getNvlmRightContour(0); }
  const uint16_t* getNvlmLeftContour() const { return const_cast<FaceIOAdapter*>(this)->getNvlmLeftContour(0); }
};

/** Write the face model as an NVF model.
 * @param[in]   fac         the face I/O adapter for the target data structure.
 * @param[in]   fileName    the desired name of the output file.
 * @return      kIOErrNone       if the file was written completed successfully.
 * @return      kIOErrFileOpen   if the file could not be opened.
 * @return      kIOErrWrite      if an error occurred while writing the file.
 */
FaceIOErr WriteNVFFaceModel(FaceIOAdapter* fac, const char* fileName);

/** Read a face model from an NVF file.
 * @param[in]       fileName    the name of the file to be read.
 * @param[in,out]   fac         the face I/O adapter for the target data structure.
 * @return      kIOErrNone       if the file was read successfully.
 * @return      kIOErrFileNotFound   if the file was not found .
 * @return      kIOErrFileOpen       if the file could not be opened.
 * @return      kIOErrRead           if an error occurred while reading the file.
 * @return      kIOErrSyntax         if a syntax error has been encountered  while reading the file.
 */
FaceIOErr ReadNVFFaceModel(const char* fileName, FaceIOAdapter* fac);

/** Read a face model from five OES files.
 * @param[in]       shape                  the name of the shape        file to be read.
 * @param[in]       ibugNumandmarks        the number of Ibug landmarks.
 * @param[in]       blendShapes            the name of the blend shapes  file to be read.
 * @param[in]       contours               the name of the contours      file to be read.
 * @param[in]       topology               the name of the topology      file to be read.
 * @param[in,out]   fac                    the face I/O adapter for the target data structure.
 * @return      kIOErrNone       if the file was read successfully.
 * @return      kIOErrFileNotFound   if the file was not found .
 * @return      kIOErrFileOpen       if the file could not be opened.
 * @return      kIOErrRead           if an error occurred while reading the file.
 * @return      kIOErrSyntax         if a syntax error has been encountered  while reading the file.
 */
FaceIOErr ReadEOSFaceModel(const char* shape, const unsigned ibugNumLandmarks, const char* blendShapes,
                           const char* contours, const char* topology, FaceIOAdapter* fac);

/** Write the face model as a JSON model.
 * @param[in]   fac         the face I/O adapter for the target data structure.
 * @param[in]   fileName    the desired name of the output file.
 *                          If NULL is supplied, it is written to the standard output.
 * @return      kIOErrNone       if the file was written completed successfully.
 * @return      kIOErrFileOpen   if the file could not be opened.
 * @return      kIOErrWrite      if an error occurred while writing the file.
 */
FaceIOErr PrintJSONFaceModel(FaceIOAdapter* fac, const char* fileName);

#endif /* __FACE_IO__ */
