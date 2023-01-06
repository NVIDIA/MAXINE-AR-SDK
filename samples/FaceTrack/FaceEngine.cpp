/*###############################################################################
#
# Copyright 2020-2021 NVIDIA Corporation
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
#include "FaceEngine.h"
#include "RenderingUtils.h"
#include <iostream>


bool CheckResult(NvCV_Status nvErr, unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
}

FaceEngine::Err FaceEngine::fitFaceModel(cv::Mat& frame) {
  FaceEngine::Err err = FaceEngine::Err::errNone;
  NvCV_Status nvErr;
  if (!frame.empty()) {
    NvCVImage fxSrcChunkyCPU;
    (void)NVWrapperForCVMat(&frame, &fxSrcChunkyCPU);
    NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, NULL);

    if (NVCV_SUCCESS != cvErr) {
      return errRun;
    }
  }

  nvErr = NvAR_Run(faceFitHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errRun);

  if (getAverageLandmarksConfidence() < confidenceThreshold) return FaceEngine::Err::errRun;

bail:
  return err;
}

NvAR_FaceMesh* FaceEngine::getFaceMesh() { return face_mesh; }

NvAR_RenderingParams* FaceEngine::getRenderingParams() { return rendering_params; }

float* FaceEngine::getShapeEigenvalues() { return shapeEigenvalues.data(); }
float* FaceEngine::getExpressionCoefficients() { return expressionCoefficients.data(); }
int FaceEngine::getNumShapeEigenvalues() {
  unsigned n = 0;
  (void)NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(ShapeEigenValueCount), &n);
  return n;
}
int FaceEngine::getNumExpressionCoefficients() {
  unsigned n = 0;
  (void)NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(ExpressionCount), &n);
  return n;
}


FaceEngine::Err FaceEngine::createFeatures(const char* modelPath, unsigned int _batchSize, unsigned int mode) {
  FaceEngine::Err err = FaceEngine::Err::errNone;

  NvCV_Status cuErr = NvAR_CudaStreamCreate(&stream);
  if (NVCV_SUCCESS != cuErr) {
    printf("Cannot create a cuda stream: %s\n", NvCV_GetErrorStringFromCode(cuErr));
    return errInitialization;
  }
  if (appMode == faceDetection) {
    err = createFaceDetectionFeature(modelPath, stream);
    if (err != Err::errNone) {
      printf("ERROR: An error has occurred while initializing Face Detection\n");
    }
  } else if (appMode == landmarkDetection) {
    err = createLandmarkDetectionFeature(modelPath, _batchSize, stream, mode);
    if (err != Err::errNone) {
      printf("ERROR: An error has occurred while initializing Landmark Detection\n");
    }
  } else if (appMode == faceMeshGeneration) {
    err = createFaceFittingFeature(modelPath, stream);
    if (err != Err::errNone) {
      printf("ERROR: An error has occurred while initializing Face Fitting\n");
    }
  }

  return err;
}

FaceEngine::Err FaceEngine::createFaceDetectionFeature(const char* modelPath, CUstream str) {
  FaceEngine::Err err = FaceEngine::Err::errNone;
  NvCV_Status nvErr;

  nvErr = NvAR_Create(NvAR_Feature_FaceDetection, &faceDetectHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errEffect);

  if (bUseOTAU && (!modelPath || !modelPath[0])) {
    nvErr = NvAR_SetString(faceDetectHandle, NvAR_Parameter_Config(ModelDir), this->fdOTAModelPath);
    BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  } else {
    nvErr = NvAR_SetString(faceDetectHandle, NvAR_Parameter_Config(ModelDir), modelPath);
    BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  }

  nvErr = NvAR_SetCudaStream(faceDetectHandle, NvAR_Parameter_Config(CUDAStream), str);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(faceDetectHandle, NvAR_Parameter_Config(Temporal), (bStabilizeFace ? -1 : 0));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_Load(faceDetectHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errInitialization);

bail:
  return err;
}

FaceEngine::Err FaceEngine::createLandmarkDetectionFeature(const char* modelPath, unsigned int _batchSize,
                                                           CUstream str, unsigned int mode) {
  FaceEngine::Err err = FaceEngine::Err::errNone;
  NvCV_Status nvErr;

  batchSize = _batchSize;
  nvErr = NvAR_Create(NvAR_Feature_LandmarkDetection, &landmarkDetectHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errEffect);

  if (bUseOTAU && (!modelPath || !modelPath[0])) {
      nvErr = NvAR_SetString(landmarkDetectHandle, NvAR_Parameter_Config(ModelDir), this->ldOTAModelPath);
      BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  }
  else {
    nvErr = NvAR_SetString(landmarkDetectHandle, NvAR_Parameter_Config(ModelDir), modelPath);
    BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  }

  nvErr = NvAR_SetCudaStream(landmarkDetectHandle, NvAR_Parameter_Config(CUDAStream), str);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(landmarkDetectHandle, NvAR_Parameter_Config(BatchSize), batchSize);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(landmarkDetectHandle, NvAR_Parameter_Config(Temporal), (bStabilizeFace ? -1 : 0));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(landmarkDetectHandle, NvAR_Parameter_Config(Landmarks_Size), numLandmarks);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(landmarkDetectHandle, NvAR_Parameter_Config(LandmarksConfidence_Size), numLandmarks);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(landmarkDetectHandle, NvAR_Parameter_Config(Mode), mode);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_Load(landmarkDetectHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errInitialization);

bail:
  return err;
}

FaceEngine::Err FaceEngine::createFaceFittingFeature(const char* modelPath, CUstream str) {
  FaceEngine::Err err = FaceEngine::Err::errNone;
  NvCV_Status nvErr;

  nvErr = NvAR_Create(NvAR_Feature_Face3DReconstruction, &faceFitHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errEffect);

  nvErr = NvAR_SetString(faceFitHandle, NvAR_Parameter_Config(ModelDir), modelPath);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetCudaStream(faceFitHandle, NvAR_Parameter_Config(CUDAStream), str);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetU32(faceFitHandle, NvAR_Parameter_Config(Landmarks_Size), numLandmarks);  // TODO: Check if nonzero??
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  if (!face_model.empty()) {
    nvErr = NvAR_SetString(faceFitHandle, NvAR_Parameter_Config(ModelName), face_model.c_str());
    BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  }

  nvErr = NvAR_Load(faceFitHandle);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errInitialization);

bail:
  return err;
}

FaceEngine::Err FaceEngine::initFeatureIOParams() {
  FaceEngine::Err err = FaceEngine::Err::errNone;

  NvCV_Status cvErr = NvCVImage_Alloc(&inputImageBuffer, input_image_width, input_image_height, NVCV_BGR, NVCV_U8,
                                      NVCV_CHUNKY, NVCV_GPU, 1);

  BAIL_IF_NVERR(cvErr, err, FaceEngine::Err::errInitialization);
  if (appMode == faceDetection) {
    err = initFaceDetectionIOParams(&inputImageBuffer);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while setting input, output parmeters for Face Detection\n");
    }
  } else if (appMode == landmarkDetection) {
    err = initLandmarkDetectionIOParams(&inputImageBuffer);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while setting input, output parmeters for Landmark Detection\n");
    }
  } else if (appMode == faceMeshGeneration) {
    err = initFaceFittingIOParams(&inputImageBuffer);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while setting input, output parmeters for Face Fitting\n");
    }
  }
  return err;

bail:
  return err;
}

FaceEngine::Err FaceEngine::initFaceDetectionIOParams(NvCVImage* inBuf) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  FaceEngine::Err err = FaceEngine::Err::errNone;

  nvErr = NvAR_SetObject(faceDetectHandle, NvAR_Parameter_Input(Image), inBuf, sizeof(NvCVImage));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  output_bbox_data.assign(25, {0.f, 0.f, 0.f, 0.f});
  output_bbox_conf_data.assign(25, 0.f);
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)output_bbox_data.size();
  output_bboxes.num_boxes = 0;
  nvErr = NvAR_SetObject(faceDetectHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(faceDetectHandle, NvAR_Parameter_Output(BoundingBoxesConfidence),
                           output_bbox_conf_data.data(), output_bboxes.max_boxes);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

bail:
  return err;
}

FaceEngine::Err FaceEngine::initLandmarkDetectionIOParams(NvCVImage* inBuf) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  FaceEngine::Err err = FaceEngine::Err::errNone;
  uint output_bbox_size;

  nvErr = NvAR_SetObject(landmarkDetectHandle, NvAR_Parameter_Input(Image), inBuf, sizeof(NvCVImage));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  unsigned int OUTPUT_SIZE_KPTS, OUTPUT_SIZE_KPTS_CONF;
  nvErr = NvAR_GetU32(landmarkDetectHandle, NvAR_Parameter_Config(Landmarks_Size), &OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_GetU32(landmarkDetectHandle, NvAR_Parameter_Config(LandmarksConfidence_Size), &OUTPUT_SIZE_KPTS_CONF);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  facial_landmarks.assign(batchSize * OUTPUT_SIZE_KPTS, {0.f, 0.f});
  facial_pose.assign(batchSize, {0.f, 0.f, 0.f, 0.f});
  facial_landmarks_confidence.assign(batchSize * OUTPUT_SIZE_KPTS_CONF, 0.f);

  nvErr = NvAR_SetObject(landmarkDetectHandle, NvAR_Parameter_Output(Landmarks), facial_landmarks.data(),
                         sizeof(NvAR_Point2f));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr =
      NvAR_SetObject(landmarkDetectHandle, NvAR_Parameter_Output(Pose), facial_pose.data(), sizeof(NvAR_Quaternion));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(landmarkDetectHandle, NvAR_Parameter_Output(LandmarksConfidence),
                           facial_landmarks_confidence.data(), batchSize * OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  output_bbox_size = batchSize;
  if (!bStabilizeFace) output_bbox_size = 25;
  output_bbox_data.assign(output_bbox_size, {0.f, 0.f, 0.f, 0.f});
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)output_bbox_size;
  output_bboxes.num_boxes = (uint8_t)output_bbox_size;
  nvErr =
      NvAR_SetObject(landmarkDetectHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

bail:
  return err;
}

FaceEngine::Err FaceEngine::initFaceFittingIOParams(NvCVImage* inBuf) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  FaceEngine::Err err = FaceEngine::Err::errNone;

  face_mesh = new NvAR_FaceMesh();

  unsigned int num_vertices, num_triangles;
  nvErr = NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(VertexCount), &num_vertices);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(TriangleCount), &num_triangles);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  face_mesh->num_vertices = num_vertices;
  face_mesh->num_triangles = num_triangles;
  m_vertices.assign(num_vertices, {0.f, 0.f, 0.f});
  m_triangles.assign(num_triangles, { 0, 0, 0 });
  face_mesh->vertices = m_vertices.data();
  face_mesh->tvi = m_triangles.data();
  rendering_params = new NvAR_RenderingParams();

  nvErr = NvAR_SetObject(faceFitHandle, NvAR_Parameter_Input(Image), inBuf, sizeof(NvCVImage));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetS32(faceFitHandle, NvAR_Parameter_Input(Width), input_image_width);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetS32(faceFitHandle, NvAR_Parameter_Input(Height), input_image_height);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  unsigned int OUTPUT_SIZE_KPTS;
  nvErr = NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(Landmarks_Size), &OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  facial_landmarks.assign(batchSize * OUTPUT_SIZE_KPTS, {0.f, 0.f});
  nvErr =
      NvAR_SetObject(faceFitHandle, NvAR_Parameter_Output(Landmarks), facial_landmarks.data(), sizeof(NvAR_Point2f));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  facial_landmarks_confidence.assign(batchSize * OUTPUT_SIZE_KPTS, 0.f);
  nvErr = NvAR_SetF32Array(faceFitHandle, NvAR_Parameter_Output(LandmarksConfidence),
                           facial_landmarks_confidence.data(), batchSize * OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  facial_pose.assign(batchSize, {0.f, 0.f, 0.f, 0.f});
  nvErr = NvAR_SetObject(faceFitHandle, NvAR_Parameter_Output(Pose), facial_pose.data(), sizeof(NvAR_Quaternion));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  output_bbox_data.assign(batchSize, {0.f, 0.f, 0.f, 0.f});
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)batchSize;
  output_bboxes.num_boxes = (uint8_t)batchSize;
  nvErr = NvAR_SetObject(faceFitHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));

  nvErr = NvAR_SetObject(faceFitHandle, NvAR_Parameter_Output(FaceMesh), face_mesh, sizeof(NvAR_FaceMesh));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_SetObject(faceFitHandle, NvAR_Parameter_Output(RenderingParams), rendering_params,
                         sizeof(NvAR_RenderingParams));
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  unsigned n;
  nvErr = NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(ShapeEigenValueCount), &n);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  shapeEigenvalues.resize(n);
  nvErr = NvAR_SetF32Array(faceFitHandle, NvAR_Parameter_Output(ShapeEigenValues), shapeEigenvalues.data(), n);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

  nvErr = NvAR_GetU32(faceFitHandle, NvAR_Parameter_Config(ExpressionCount), &n);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);
  expressionCoefficients.resize(n);
  nvErr = NvAR_SetF32Array(faceFitHandle, NvAR_Parameter_Output(ExpressionCoefficients), expressionCoefficients.data(), n);
  BAIL_IF_NVERR(nvErr, err, FaceEngine::Err::errParameter);

bail:
  return err;
}

void FaceEngine::destroyFeatures() {
  if (stream) {
    NvAR_CudaStreamDestroy(stream);
    stream = 0;
  }
  releaseFeatureIOParams();
  destroyFaceDetectionFeature();
  destroyLandmarkDetectionFeature();
  destroyFaceFittingFeature();
}

void FaceEngine::destroyFaceDetectionFeature() {
  if (faceDetectHandle) {
    (void)NvAR_Destroy(faceDetectHandle);
    faceDetectHandle = nullptr;
  }
}
void FaceEngine::destroyLandmarkDetectionFeature() {
  if (landmarkDetectHandle) {
    (void)NvAR_Destroy(landmarkDetectHandle);
    landmarkDetectHandle = nullptr;
  }
}
void FaceEngine::destroyFaceFittingFeature() {
  if (faceFitHandle) {
    (void)NvAR_Destroy(faceFitHandle);
    faceFitHandle = nullptr;
  }
}

void FaceEngine::releaseFeatureIOParams() {
  releaseFaceDetectionIOParams();
  releaseLandmarkDetectionIOParams();
  releaseFaceFittingIOParams();
}

void FaceEngine::releaseFaceDetectionIOParams() {
  NvCVImage_Dealloc(&inputImageBuffer);
  if (!output_bbox_data.empty()) output_bbox_data.clear();
  if (!output_bbox_conf_data.empty()) output_bbox_conf_data.clear();
}

void FaceEngine::releaseLandmarkDetectionIOParams() {
  NvCVImage_Dealloc(&inputImageBuffer);
  if (!output_bbox_data.empty()) output_bbox_data.clear();
  if (!facial_landmarks.empty()) facial_landmarks.clear();
  if (!facial_pose.empty()) facial_pose.clear();
  if (!facial_landmarks_confidence.empty()) facial_landmarks_confidence.clear();
}

void FaceEngine::releaseFaceFittingIOParams() {
  if (!output_bbox_data.empty()) output_bbox_data.clear();
  if (!facial_landmarks.empty()) facial_landmarks.clear();
  if (!facial_landmarks_confidence.empty()) facial_landmarks_confidence.clear();
  if (!facial_pose.empty()) facial_pose.clear();
  NvCVImage_Dealloc(&inputImageBuffer);
  if (rendering_params) {
    delete rendering_params;
    rendering_params = nullptr;
  }
  if (face_mesh) {
    delete face_mesh;
    face_mesh = nullptr;
  }
}

NvCV_Status FaceEngine::findFaceBoxes(unsigned &num_boxes) {
  NvCV_Status nvErr = NvAR_Run(faceDetectHandle);
  num_boxes = (unsigned)output_bboxes.num_boxes;
  return nvErr;
}

NvAR_Rect* FaceEngine::getLargestBox() {
  NvAR_Rect *box, *bigBox, *lastBox;
  float maxArea, area;
  for (lastBox = (box = &output_bboxes.boxes[0]) + output_bboxes.num_boxes, bigBox = nullptr, maxArea = 0;
       box != lastBox; ++box) {
    if (maxArea < (area = box->width * box->height)) {
      maxArea = area;
      bigBox = box;
    }
  }
  return bigBox;
}

NvAR_BBoxes* FaceEngine::getBoundingBoxes() { return &output_bboxes; }

void FaceEngine::enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant) {
  NvAR_Vector2f size = {box.width * .5f, box.height * .5f};
  NvAR_Point2f center = {box.x + size.x, box.y + size.y};
  float t;

  size.x *= (1.f + enlarge);
  size.y *= (1.f + enlarge);

  if (!(FLAG_variant & 1)) /* Default: enforce square bounding box */
  {
    if (size.x < size.y) /* Make square */
      size.x = size.y;
    else
      size.y = size.x;
  }

  if (center.x < size.x) /* Shift box into image left-right */
    center.x = size.x;
  else if (center.x > (t = input_image_width - size.x))
    center.x = t;

  if (center.y < size.y) /* Shift box into image up-down */
    center.y = size.y;
  else if (center.y > (t = input_image_height - size.y))
    center.y = t;

  // TODO: Above we assume that the box is smaller than the image.

  box.width = roundf(size.x * 2.f); /* Integral box */
  box.height = roundf(size.y * 2.f);
  box.x = roundf(center.x - box.width * .5f);
  box.y = roundf(center.y - box.height * .5f);
}

/**
 * Perturb a bounding box.
 * @param[in,out] ran the random number generator.
 * @param[in] minMag inset magnitude of noise.
 * @param[in] maxMag outset magnitude of noise.
 * @param[in] cleanBox the clean input box.
 * @param[out] noisyBox the perturbed output box.
 */
// TOD0: Fix the 8 bounding boxes instead if random jiggling
void FaceEngine::jiggleBox(std::mt19937& ran, float minMag, float maxMag, const NvAR_Rect& cleanBox,
                           NvAR_Rect& jitteredBox) {
  if (0.f == minMag && 0.f == maxMag) {
    jitteredBox = cleanBox;
    return;
  }

  std::uniform_real_distribution<float> uRand(minMag, maxMag);
#if 1  // jitter both sides
  jitteredBox.x = cleanBox.x - uRand(ran);
  jitteredBox.y = cleanBox.y - uRand(ran);
  jitteredBox.width = cleanBox.width + cleanBox.x + uRand(ran) - jitteredBox.x;
  jitteredBox.height = cleanBox.height + cleanBox.y + uRand(ran) - jitteredBox.y;
#elif 0  // jitter only the right side
  noisyBox.x = cleanBox.x;
  noisyBox.y = cleanBox.y;
  noisyBox.width = cleanBox.width + cleanBox.x + uRand(ran) - noisyBox.x;
  noisyBox.height = cleanBox.height + cleanBox.y + uRand(ran) - noisyBox.y;
#elif 1  // jitter only the location
  noisyBox.x = cleanBox.x - uRand(ran);
  noisyBox.y = cleanBox.y - uRand(ran);
  noisyBox.width = cleanBox.width;
  noisyBox.height = cleanBox.height;
#endif   // None of these makes a significant difference
}

#ifdef UNUSED
/** Intersect a rectangle with an image.
 * @param[in]   srcRect   the source rectangle.
 * @param[in]   src       the source image.
 * @param[out]  clipRect  the source rectangle clipped to the source image.
 * @return      true      if the rectangle was clipped,
 *              false     if the rectangle did not need to be clipped.
 */
static bool IntersectRectWithImage(const cv::Rect& srcRect, const cv::Mat& src, cv::Rect& clipRect) {
  bool result = 0;  // unclipped
  cv::Point2i rect[2] = {{srcRect.x, srcRect.y}, {srcRect.x + srcRect.width, srcRect.y + srcRect.height}};
  if (rect[0].x < 0) {
    result = 1;
    rect[0].x = 0;
  }  // clipped
  if (rect[0].y < 0) {
    result = 1;
    rect[0].y = 0;
  }  // clipped
  if (rect[1].x > src.cols) {
    result = 1;
    rect[1].x = src.cols;
  }  // clipped
  if (rect[1].y > src.rows) {
    result = 1;
    rect[1].y = src.rows;
  }  // clipped
  clipRect.x = rect[0].x;
  clipRect.y = rect[0].y;
  clipRect.width = rect[1].x - rect[0].x;
  clipRect.height = rect[1].y - rect[0].y;
  return result;
}
#endif // UNUSED

void FaceEngine::DrawPose(const cv::Mat& src, const NvAR_Quaternion* pose) const {
  const float vector_scale = 50.0f;
  const int thickness = 2;
  float rot_mat[3][3];
  set_rotation_from_quaternion(pose, rot_mat[0]);
  const auto avg_landmark_pos = GetAverageLandmarkPositionInGlSpace();

  cv::Point p0 = {static_cast<int>(avg_landmark_pos[0]), static_cast<int>(avg_landmark_pos[1])};
  cv::Point px = p0 + cv::Point(std::lround(rot_mat[0][0] * vector_scale), std::lround(rot_mat[0][1] * vector_scale));
  cv::Point py = p0 + cv::Point(std::lround(rot_mat[1][0] * vector_scale), std::lround(rot_mat[1][1] * vector_scale));
  cv::Point pz = p0 + cv::Point(std::lround(rot_mat[2][0] * vector_scale), std::lround(rot_mat[2][1] * vector_scale));

  // Convert from OpenGL to OpenCV space
  p0.y = src.rows - 1 - p0.y;
  px.y = src.rows - 1 - px.y;
  py.y = src.rows - 1 - py.y;
  pz.y = src.rows - 1 - pz.y;

  cv::line(src, p0, px, CV_RGB(255, 0, 0), thickness);
  cv::line(src, p0, py, CV_RGB(0, 255, 0), thickness);
  cv::line(src, p0, pz, CV_RGB(0, 0, 255), thickness);
}

FaceEngine::Err FaceEngine::findLandmarks() {
  NvCV_Status nvErr;

  nvErr = NvAR_Run(landmarkDetectHandle);
  if (NVCV_SUCCESS != nvErr) {
    return FaceEngine::Err::errRun;
  }

  float avg_confidence = getAverageLandmarksConfidence();

  if (avg_confidence < confidenceThreshold) {
    return FaceEngine::Err::errNoFaceDetected;
  } else {
    average_poses(getPose(), batchSize);
    NvAR_Point2f *pt, *endPt;
    int i = 0;
    for (endPt = (pt = getLandmarks()) + numLandmarks; pt != endPt; ++pt, i += 2) {
      for (int j = 1; j < batchSize; j++) {
        pt->x += pt[j * numLandmarks].x;
        pt->y += pt[j * numLandmarks].y;
      }
      // average batch of inferences to generate final result landmark points
      pt->x /= batchSize;
      pt->y /= batchSize;
    }
  }
  return FaceEngine::Err::errNone;
}

NvAR_Point2f* FaceEngine::getLandmarks() { return facial_landmarks.data(); }

float* FaceEngine::getLandmarksConfidence() { return facial_landmarks_confidence.data(); }

std::array<float, 2> FaceEngine::GetAverageLandmarkPositionInGlSpace() const {
  std::array<float, 2> res = { 0.0f, 0.0f };
  for (const auto& landmark : facial_landmarks) {
    res[0] += landmark.x;
    res[1] += landmark.y;
  }
  res[0] /= facial_landmarks.size();
  res[1] /= facial_landmarks.size();
  res[1] = input_image_height - res[1]; //Convert y coordinate from CV to GL space
  return res;
}

float FaceEngine::getAverageLandmarksConfidence() {
  float average_confidence = 0.0f;
  float* keypoints_landmarks_confidence = getLandmarksConfidence();
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < numLandmarks; j++) {
      average_confidence += keypoints_landmarks_confidence[i * numLandmarks + j];
    }
  }
  average_confidence /= batchSize * numLandmarks;
  return average_confidence;
}

NvAR_Quaternion* FaceEngine::getPose() { return facial_pose.data(); }

FaceEngine::Err FaceEngine::findLargestFaceBox(NvAR_Rect& faceBox, int variant) {
  NvAR_Rect* pFaceBox;
  unsigned num_boxes;

  NvCV_Status nv_err = findFaceBoxes(num_boxes);
  if (NVCV_SUCCESS != nv_err) {
      return FaceEngine::Err::errRun;
    }

  if (num_boxes >= 1) {
    pFaceBox = getLargestBox();
    if (nullptr == pFaceBox) {
      faceBox.x = faceBox.y = faceBox.width = faceBox.height = 0.0f;
    } else {
      faceBox = *pFaceBox;
    }
    enlargeAndSquarifyImageBox(.2f, faceBox, variant);
  }
  else return FaceEngine::Err::errNoFaceDetected;

  return FaceEngine::Err::errNone;
}

FaceEngine::Err FaceEngine::acquireFaceBox(cv::Mat& src, NvAR_Rect& faceBox, int variant) {
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return FaceEngine::Err::errRun;
  }
  FaceEngine::Err nv_err = findLargestFaceBox(faceBox, variant);
  
  return nv_err;
}

FaceEngine::Err FaceEngine::acquireFaceBoxAndLandmarks(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Rect& faceBox, int /*variant*/) {
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return FaceEngine::Err::errRun;
  }

  FaceEngine::Err nv_err = findLandmarks();
  
  if(nv_err == FaceEngine::Err::errNone){
    faceBox = output_bboxes.boxes[0];
    memcpy(refMarks, getLandmarks(), sizeof(NvAR_Point2f) * numLandmarks);
  }
  
  return nv_err;
}

void FaceEngine::setFaceStabilization(bool _bStabilizeFace) { bStabilizeFace = _bStabilizeFace; }

FaceEngine::Err FaceEngine::setNumLandmarks(int n) {
  FaceEngine::Err err = errNone;
#ifdef _WIN32 // Keep Updated Thresholds for Windows Only
  for (auto const& info : LANDMARKS_INFO) 
#else
  for (auto const& info : LANDMARKS_INFO_UNIX)
#endif
  {
    if (n == info.numPoints) {
      numLandmarks = info.numPoints;
      confidenceThreshold = info.confidence_threshold;
      return err;
    }
  }    
  err = errGeneral;
  return err;
}

void FaceEngine::setAppMode(FaceEngine::mode _mode) { appMode = _mode; }
