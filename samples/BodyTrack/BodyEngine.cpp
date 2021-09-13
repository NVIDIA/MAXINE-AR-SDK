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
#include "BodyEngine.h"
#include <iostream>

bool CheckResult(NvCV_Status nvErr, unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
}

BodyEngine::Err BodyEngine::createFeatures(const char* modelPath, unsigned int _batchSize) {
  BodyEngine::Err err = BodyEngine::Err::errNone;

  NvCV_Status cuErr = NvAR_CudaStreamCreate(&stream);
  if (NVCV_SUCCESS != cuErr) {
    printf("Cannot create a cuda stream: %s\n", NvCV_GetErrorStringFromCode(cuErr));
    return errInitialization;
  }
  if (appMode == bodyDetection) {
    err = createBodyDetectionFeature(modelPath, stream);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while initializing Body Detection\n");
    }
  }
  else if (appMode == keyPointDetection) {
    err = createKeyPointDetectionFeature(modelPath, _batchSize, stream);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while initializing KeyPoint Detection\n");
    }
  }
  return err;
}

BodyEngine::Err BodyEngine::createBodyDetectionFeature(const char* modelPath, CUstream str) {
  BodyEngine::Err err = BodyEngine::Err::errNone;
  NvCV_Status nvErr;

  nvErr = NvAR_Create(NvAR_Feature_BodyDetection, &bodyDetectHandle);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errEffect);

  if (bUseOTAU && (!modelPath || !modelPath[0])) {
    nvErr = NvAR_SetString(bodyDetectHandle, NvAR_Parameter_Config(ModelDir), this->bdOTAModelPath);
    BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);
  } else {
    nvErr = NvAR_SetString(bodyDetectHandle, NvAR_Parameter_Config(ModelDir), modelPath);
    BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);
  }

  nvErr = NvAR_SetCudaStream(bodyDetectHandle, NvAR_Parameter_Config(CUDAStream), str);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetU32(bodyDetectHandle, NvAR_Parameter_Config(Temporal), bStabilizeBody);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_Load(bodyDetectHandle);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errInitialization);

bail:
  return err;
}

BodyEngine::Err BodyEngine::createKeyPointDetectionFeature(const char* modelPath, unsigned int _batchSize,
                                                           CUstream str) {
  BodyEngine::Err err = BodyEngine::Err::errNone;
  NvCV_Status nvErr;

  batchSize = _batchSize;
  nvErr = NvAR_Create(NvAR_Feature_BodyPoseEstimation, &keyPointDetectHandle);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errEffect);

  if (bUseOTAU && (!modelPath || !modelPath[0])) {
    nvErr = NvAR_SetString(keyPointDetectHandle, NvAR_Parameter_Config(ModelDir), this->ldOTAModelPath);
    BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);
  }
  else {
    nvErr = NvAR_SetString(keyPointDetectHandle, NvAR_Parameter_Config(ModelDir), modelPath);
    BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);
  }

  nvErr = NvAR_SetCudaStream(keyPointDetectHandle, NvAR_Parameter_Config(CUDAStream), str);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetU32(keyPointDetectHandle, NvAR_Parameter_Config(BatchSize), batchSize);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetU32(keyPointDetectHandle, NvAR_Parameter_Config(Mode), nvARMode);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetU32(keyPointDetectHandle, NvAR_Parameter_Config(Temporal), bStabilizeBody);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetF32(keyPointDetectHandle, NvAR_Parameter_Config(FocalLength), bFocalLength);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetF32(keyPointDetectHandle, NvAR_Parameter_Config(UseCudaGraph), bUseCudaGraph);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_Load(keyPointDetectHandle);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errInitialization);

bail:
  return err;
}

BodyEngine::Err BodyEngine::initFeatureIOParams() {
  BodyEngine::Err err = BodyEngine::Err::errNone;

  NvCV_Status cvErr = NvCVImage_Alloc(&inputImageBuffer, input_image_width, input_image_height, NVCV_BGR, NVCV_U8,
                                      NVCV_CHUNKY, NVCV_GPU, 1);

  BAIL_IF_CVERR(cvErr, err, BodyEngine::Err::errInitialization);

  if (appMode == bodyDetection) {
    err = initBodyDetectionIOParams(&inputImageBuffer);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while setting input, output parmeters for Body Detection\n");
    }
  }
  else if (appMode == keyPointDetection) {
    err = initKeyPointDetectionIOParams(&inputImageBuffer);
    if (err != Err::errNone) {
      printf("ERROR: An error has occured while setting input, output parmeters for KeyPoint Detection\n");
    }
  }
  return err;

bail:
  return err;
}

BodyEngine::Err BodyEngine::initBodyDetectionIOParams(NvCVImage* inBuf) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  BodyEngine::Err err = BodyEngine::Err::errNone;

  nvErr = NvAR_SetObject(bodyDetectHandle, NvAR_Parameter_Input(Image), inBuf, sizeof(NvCVImage));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  output_bbox_data.assign(25, {0.f, 0.f, 0.f, 0.f});
  output_bbox_conf_data.assign(25, 0.f);
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)output_bbox_data.size();
  output_bboxes.num_boxes = 0;
  nvErr = NvAR_SetObject(bodyDetectHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(bodyDetectHandle, NvAR_Parameter_Output(BoundingBoxesConfidence),
                           output_bbox_conf_data.data(), output_bboxes.max_boxes);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

bail:
  return err;
}

BodyEngine::Err BodyEngine::initKeyPointDetectionIOParams(NvCVImage* inBuf) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  BodyEngine::Err err = BodyEngine::Err::errNone;
  uint output_bbox_size;

  nvErr = NvAR_SetObject(keyPointDetectHandle, NvAR_Parameter_Input(Image), inBuf, sizeof(NvCVImage));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_GetU32(keyPointDetectHandle, NvAR_Parameter_Config(NumKeyPoints), &numKeyPoints);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  keypoints.assign(batchSize * numKeyPoints, {0.f, 0.f});
  keypoints3D.assign(batchSize * numKeyPoints, {0.f, 0.f, 0.f});
  jointAngles.assign(batchSize * numKeyPoints, {0.f, 0.f, 0.f, 1.f});
  keypoints_confidence.assign(batchSize * numKeyPoints, 0.f);
  referencePose.assign(numKeyPoints, {0.f, 0.f, 0.f});

  const void* pReferencePose;
  nvErr = NvAR_GetObject(keyPointDetectHandle, NvAR_Parameter_Config(ReferencePose), &pReferencePose,
                         sizeof(NvAR_Point3f));
  memcpy(referencePose.data(), pReferencePose, sizeof(NvAR_Point3f) * numKeyPoints);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetObject(keyPointDetectHandle, NvAR_Parameter_Output(KeyPoints), keypoints.data(),
                         sizeof(NvAR_Point2f));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetObject(keyPointDetectHandle, NvAR_Parameter_Output(KeyPoints3D), keypoints3D.data(),
                                                 sizeof(NvAR_Point3f));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetObject(keyPointDetectHandle, NvAR_Parameter_Output(JointAngles), jointAngles.data(),
                         sizeof(NvAR_Quaternion));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(keyPointDetectHandle, NvAR_Parameter_Output(KeyPointsConfidence),
                           keypoints_confidence.data(), batchSize * numKeyPoints);
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

  output_bbox_size = batchSize;
  if (!bStabilizeBody) output_bbox_size = 25;
  output_bbox_data.assign(output_bbox_size, {0.f, 0.f, 0.f, 0.f});
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)output_bbox_size;
  output_bboxes.num_boxes = (uint8_t)output_bbox_size;
  nvErr =
      NvAR_SetObject(keyPointDetectHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));
  BAIL_IF_CVERR(nvErr, err, BodyEngine::Err::errParameter);

bail:
  return err;
}

void BodyEngine::destroyFeatures() {
  if (stream) {
    NvAR_CudaStreamDestroy(stream);
    stream = 0;
  }
  releaseFeatureIOParams();
  destroyBodyDetectionFeature();
  destroyKeyPointDetectionFeature();
}

void BodyEngine::destroyBodyDetectionFeature() {
  if (bodyDetectHandle) {
    (void)NvAR_Destroy(bodyDetectHandle);
    bodyDetectHandle = nullptr;
  }
}
void BodyEngine::destroyKeyPointDetectionFeature() {
  if (keyPointDetectHandle) {
    (void)NvAR_Destroy(keyPointDetectHandle);
    keyPointDetectHandle = nullptr;
  }
}

void BodyEngine::releaseFeatureIOParams() {
  releaseBodyDetectionIOParams();
  releaseKeyPointDetectionIOParams();
}

void BodyEngine::releaseBodyDetectionIOParams() {
  NvCVImage_Dealloc(&inputImageBuffer);
  if (!output_bbox_data.empty()) output_bbox_data.clear();
  if (!output_bbox_conf_data.empty()) output_bbox_conf_data.clear();
}

void BodyEngine::releaseKeyPointDetectionIOParams() {
  NvCVImage_Dealloc(&inputImageBuffer);
  if (!output_bbox_data.empty()) output_bbox_data.clear();
  if (!keypoints.empty()) keypoints.clear();
  if (!keypoints3D.empty()) keypoints3D.clear();
  if (!jointAngles.empty()) jointAngles.clear();
  if (!keypoints_confidence.empty()) keypoints_confidence.clear();
}

unsigned BodyEngine::findBodyBoxes() {
  NvCV_Status nvErr = NvAR_Run(bodyDetectHandle);
  if (NVCV_SUCCESS != nvErr) return 0;
  return (unsigned)output_bboxes.num_boxes;
}

NvAR_Rect* BodyEngine::getLargestBox() {
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

NvAR_BBoxes* BodyEngine::getBoundingBoxes() { return &output_bboxes; }

void BodyEngine::enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant) {
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

 NvCV_Status BodyEngine::findKeyPoints() {
   NvCV_Status nvErr;
#ifdef DEBUG_PERF_RUNTIME
   auto start = std::chrono::high_resolution_clock::now();
#endif
   nvErr = NvAR_Run(keyPointDetectHandle);
   if (NVCV_SUCCESS != nvErr) {
     return nvErr;
   }
#ifdef DEBUG_PERF_RUNTIME
   auto end = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   std::cout << "[bodypose] > NvAR_Run(keyPointDetectHandle): " << duration.count() << " microseconds" << std::endl;
#endif

   if (getAverageKeyPointsConfidence() < confidenceThreshold) {
     return NVCV_ERR_GENERAL;
   } else {
     NvAR_Point2f *pt, *endPt;
     int i = 0;
     for (endPt = (pt = getKeyPoints()) + numKeyPoints; pt != endPt; ++pt, i += 2) {
       for (int j = 1; j < batchSize; j++) {
         pt->x += pt[j * numKeyPoints].x;
         pt->y += pt[j * numKeyPoints].y;
       }
       // average batch of inferences to generate final result keypoints
       pt->x /= batchSize;
       pt->y /= batchSize;
     }
   }
#ifdef DEBUG_PERF_RUNTIME
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "[bodypose] inside findKeyPoints(): " << duration.count() << " microseconds" << std::endl;
#endif
   return NVCV_SUCCESS;
 }

NvAR_Point2f* BodyEngine::getKeyPoints() { return keypoints.data(); }

NvAR_Point3f* BodyEngine::getKeyPoints3D() { return keypoints3D.data(); }

NvAR_Quaternion* BodyEngine::getJointAngles() { return jointAngles.data(); }

 float* BodyEngine::getKeyPointsConfidence() { return keypoints_confidence.data(); }

 float BodyEngine::getAverageKeyPointsConfidence() {
   float average_confidence = 0.0f;
   float* keypoints_confidence_all = getKeyPointsConfidence();
   for (int i = 0; i < batchSize; i++) {
     for (unsigned int j = 0; j < numKeyPoints; j++) {
       average_confidence += keypoints_confidence_all[i * numKeyPoints + j];
     }
   }
   average_confidence /= batchSize * numKeyPoints;
   return average_confidence;
 }

unsigned BodyEngine::findLargestBodyBox(NvAR_Rect& bodyBox, int /*variant*/) {
  unsigned n;
  NvAR_Rect* pBodyBox;

  n = findBodyBoxes();
  if (n >= 1) {
    pBodyBox = getLargestBox();
    if (nullptr == pBodyBox) {
      bodyBox.x = bodyBox.y = bodyBox.width = bodyBox.height = 0.0f;
    } else {
      bodyBox = *pBodyBox;
    }
    //enlargeAndSquarifyImageBox(.2f, bodyBox, variant);
  }
  return n;
}

unsigned BodyEngine::acquireBodyBox(cv::Mat& src, NvAR_Rect& bodyBox, int variant) {
  unsigned n = 0;
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return n;
  }

  n = findLargestBodyBox(bodyBox, variant);
  return n;
}

unsigned BodyEngine::acquireBodyBoxAndKeyPoints(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Point3f* refKeyPoints3D,
                                                NvAR_Quaternion* refJointAngles, NvAR_Rect& bodyBox, int /*variant*/) {
  unsigned n = 0;
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return n;
  }
#ifdef DEBUG_PERF_RUNTIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  if (findKeyPoints() != NVCV_SUCCESS) return 0;
  bodyBox = output_bboxes.boxes[0];
  n = 1;
#ifdef DEBUG_PERF_RUNTIME
  auto start2 = std::chrono::high_resolution_clock::now();
#endif
  memcpy(refMarks, getKeyPoints(), sizeof(NvAR_Point2f) * numKeyPoints);
  memcpy(refKeyPoints3D, getKeyPoints3D(), sizeof(NvAR_Point3f) * numKeyPoints);
  memcpy(refJointAngles, getJointAngles(), sizeof(NvAR_Quaternion) * numKeyPoints);

#ifdef DEBUG_PERF_RUNTIME
  auto end = std::chrono::high_resolution_clock::now();
  auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(start2 - start);
  std::cout << "[bodypose] run findKeyPoints(): " << duration3.count() << " microseconds" << std::endl;
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start2);
  std::cout << "[bodypose] keypoint copy time: " << duration2.count() << " microseconds" << std::endl;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "[bodypose] end-to-end time: " << duration.count() << " microseconds" << std::endl;
#endif
  return n;
}

void BodyEngine::setBodyStabilization(bool _bStabilizeBody) { bStabilizeBody = _bStabilizeBody; }

void BodyEngine::setMode(int _mode) { nvARMode = _mode; }

void BodyEngine::setFocalLength(float _bFocalLength) { bFocalLength = _bFocalLength; }

void BodyEngine::useCudaGraph(bool _bUseCudaGraph) { bUseCudaGraph = _bUseCudaGraph; }

void BodyEngine::setAppMode(BodyEngine::mode _mode) { appMode = _mode; }
