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
#ifndef NvAR_DEFS_H
#define NvAR_DEFS_H

#include <stddef.h>
#include <stdint.h>
#include <nvCVImage.h>
#include <nvCVStatus.h>

#ifdef _WIN32
  #ifdef NVAR_API_EXPORT
    #define NvAR_API __declspec(dllexport) __cdecl
  #else
    #define NvAR_API
  #endif
#else
  #define NvAR_API
#endif // OS dependencies

// TODO: Change the representation to x,y,z instead of array
typedef struct NvAR_Vector3f
{
  float vec[3];
} NvAR_Vector3f;

typedef struct NvAR_Vector3u16 {
  unsigned short vec[3];
} NvAR_Vector3u16;

typedef struct NvAR_Rect {
  float x, y, width, height;
} NvAR_Rect;

typedef struct NvAR_BBoxes {
  NvAR_Rect *boxes;
  uint8_t num_boxes;
  uint8_t max_boxes;
} NvAR_BBoxes;

typedef struct NvAR_FaceMesh {
  NvAR_Vector3f *vertices;  ///< Mesh 3D vertex positions.
  size_t num_vertices;
  NvAR_Vector3u16 *tvi;  ///< Mesh triangle's vertex indices
  size_t num_tri_idx;
} NvAR_FaceMesh;

typedef struct NvAR_Frustum {
  float left;
  float right;
  float bottom;
  float top;
} NvAR_Frustum;

typedef struct NvAR_Quaternion {
  float x, y, z, w;
} NvAR_Quaternion;

typedef struct NvAR_Point2f {
  float x, y;
} NvAR_Point2f;

typedef struct NvAR_Point3f {
  float x, y, z;
} NvAR_Point3f;

typedef struct NvAR_Vector2f {
  float x, y;
} NvAR_Vector2f;

typedef struct NvAR_RenderingParams {
  NvAR_Frustum frustum;
  NvAR_Quaternion rotation;
  NvAR_Vector3f translation;
} NvAR_RenderingParams;

// Parameters provided by client application
typedef const char* NvAR_FeatureID;
#define NvAR_Feature_FaceDetection "FaceDetection"
#define NvAR_Feature_LandmarkDetection "LandmarkDetection"
#define NvAR_Feature_Face3DReconstruction "Face3DReconstruction"
#define NvAR_Feature_BodyDetection "BodyDetection"
#define NvAR_Feature_BodyPoseEstimation "BodyPoseEstimation"

#define NvAR_Parameter_Input(Name) "NvAR_Parameter_Input_" #Name
#define NvAR_Parameter_Output(Name) "NvAR_Parameter_Output_" #Name
#define NvAR_Parameter_Config(Name) "NvAR_Parameter_Config_" #Name
#define NvAR_Parameter_InOut(Name) "NvAR_Parameter_InOut_" #Name

/*
Parameters supported by each NvAR_FeatureID

*******NvAR_Feature_FaceDetection*******
Config:
NvAR_Parameter_Config(FeatureDescription)
NvAR_Parameter_Config(CUDAStream)
NvAR_Parameter_Config(TRTModelDir)
NvAR_Parameter_Config(Temporal)

Input:
NvAR_Parameter_Input(Image)

Output:
NvAR_Parameter_Output(BoundingBoxes)
NvAR_Parameter_Output(BoundingBoxesConfidence) - OPTIONAL

*******NvAR_Feature_LandmarkDetection*******
Config:
NvAR_Parameter_Config(FeatureDescription)
NvAR_Parameter_Config(CUDAStream)
NvAR_Parameter_Config(ModelDir)
NvAR_Parameter_Config(BatchSize)
NvAR_Parameter_Config(Landmarks_Size)
NvAR_Parameter_Config(LandmarksConfidence_Size)
NvAR_Parameter_Config(Temporal)

Input:
NvAR_Parameter_Input(Image)
NvAR_Parameter_Input(BoundingBoxes) - OPTIONAL

Output:
NvAR_Parameter_Output(BoundingBoxes) - OPTIONAL
NvAR_Parameter_Output(Landmarks)
NvAR_Parameter_Output(Pose) - OPTIONAL
NvAR_Parameter_Output(LandmarksConfidence) - OPTIONAL

*******NvAR_Feature_Face3DReconstruction*******
Config:
NvAR_Parameter_Config(FeatureDescription)
NvAR_Parameter_Config(ModelDir)
NvAR_Parameter_Config(Landmarks_Size)
NvAR_Parameter_Config(CUDAStream) -OPTIONAL
NvAR_Parameter_Config(Temporal) - OPTIONAL
NvAR_Parameter_Config(ModelName) - OPTIONAL
NvAR_Parameter_Config(GPU) - OPTIONAL
NvAR_Parameter_Config(VertexCount) - QUERY
NvAR_Parameter_Config(TriangleCount) - QUERY
NvAR_Parameter_Config(ExpressionCount) - QUERY
NvAR_Parameter_Config(ShapeEigenValueCount) - QUERY

Input:
NvAR_Parameter_Input(Width)
NvAR_Parameter_Input(Height)
NvAR_Parameter_Input(Image) - OPTIONAL
NvAR_Parameter_Input(Landmarks) - OPTIONAL

Output:
NvAR_Parameter_Output(FaceMesh)
NvAR_Parameter_Output(RenderingParams)
NvAR_Parameter_Output(BoundingBoxes) - OPTIONAL
NvAR_Parameter_Output(BoundingBoxesConfidence) - OPTIONAL
NvAR_Parameter_Output(Landmarks) - OPTIONAL
NvAR_Parameter_Output(Pose) - OPTIONAL
NvAR_Parameter_Output(LandmarksConfidence) - OPTIONAL
NvAR_Parameter_Output(ExpressionCoefficients) - OPTIONAL
NvAR_Parameter_Output(ShapeEigenValues) - OPTIONAL
*/


#endif  // NvAR_DEFS_H
