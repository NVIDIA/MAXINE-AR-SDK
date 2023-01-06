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
#ifndef __BODY_ENGINE__
#define __BODY_ENGINE__

#include <random>
#include <chrono>
#include "nvAR.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"
// #include "FeatureVertexName.h"
#define FITBODY_PRIVATE

class KalmanFilter1D {
 private:
  float Q_;          // Covariance of the process noise
  float xhat_;       // Current prediction
  float xhatminus_;  // Previous prediction
  float P_;          // Estimated accuracy of xhat_
  float Pminus_;     // Previous P_
  float K_;          // Kalman gain
  float R_;          // Covariance of the observation noise
  bool bFirstUse_;

 public:
  KalmanFilter1D() { reset(); }

  KalmanFilter1D(float Q, float R) { reset(Q, R); }

  void reset() {
    R_ = 0.005f * 0.005f;
    Q_ = 1e-5f;
    xhat_ = 0.0f;
    xhatminus_ = 0.0f;
    P_ = 1;
    bFirstUse_ = true;
    Pminus_ = 0.0f;
    K_ = 0.0f;
  }

  void reset(float Q, float R) {
    reset();
    Q_ = Q;
    R_ = R;
  }

  float update(float val) {
    if (bFirstUse_) {
      xhat_ = val;
      bFirstUse_ = false;
    }

    xhatminus_ = xhat_;
    Pminus_ = P_ + Q_;
    K_ = Pminus_ / (Pminus_ + R_);
    xhat_ = xhatminus_ + K_ * (val - xhatminus_);
    P_ = (1 - K_) * Pminus_;

    return xhat_;
  }
};

bool CheckResult(NvCV_Status nvErr, unsigned line);

#define BAIL_IF_ERR(err)                 \
do {                                     \
    if (0 != (err)) {                    \
      goto bail;                         \
    }                                    \
  } while (0)

#define BAIL_IF_CVERR(nvErr, err, code)  \
  do {                                   \
    if (!CheckResult(nvErr, __LINE__)) { \
      err = code;                        \
      goto bail;                         \
    }                                    \
  } while (0)

typedef struct KeyPointsProperties {
   int numPoints;
   float confidence_threshold;
}KeyPointsProperties;

// This default focal length matches a logitech webcam
static const float FOCAL_LENGTH_DEFAULT = 800.f;

/********************************************************************************
 * BodyEngine
 ********************************************************************************/

class BodyEngine {
 public:
  enum Err { errNone, errGeneral, errRun, errInitialization, errRead, errEffect, errParameter };
  int input_image_width, input_image_height, input_image_pitch;
  
  void setInputImageWidth(int width) { input_image_width = width; }
  void setInputImageHeight(int height) { input_image_height = height; }
  int getInputImageWidth() { return input_image_width; }
  int getInputImageHeight() { return input_image_height; }
  int getInputImagePitch() { return input_image_pitch = input_image_width * 3 * sizeof(unsigned char); }
  void setBodyModel(const char *bodyModel) { body_model = bodyModel; }

  Err createFeatures(const char* modelPath, unsigned int _batchSize = 1);
  Err createBodyDetectionFeature(const char* modelPath, CUstream stream);
  Err createKeyPointDetectionFeature(const char* modelPath, unsigned int batchSize, CUstream stream);
  void destroyFeatures();
  void destroyBodyDetectionFeature();
  void destroyKeyPointDetectionFeature();
  Err initFeatureIOParams();
  Err initBodyDetectionIOParams(NvCVImage* _inputImageBuffer);
  Err initKeyPointDetectionIOParams(NvCVImage* _inputImageBuffer);
  void releaseFeatureIOParams();
  void releaseBodyDetectionIOParams();
  void releaseKeyPointDetectionIOParams();

  unsigned findBodyBoxes();
  NvAR_Rect* getLargestBox();
  NvCV_Status findKeyPoints();
  NvAR_BBoxes* getBoundingBoxes();
  NvAR_Point2f* getKeyPoints();
  NvAR_Point3f* getKeyPoints3D();
  NvAR_Quaternion* getJointAngles();
  NvAR_BBoxes* getBBoxes();
#if NV_MULTI_OBJECT_TRACKER
  NvAR_TrackingBBoxes* getTrackingBBoxes();
#endif
  float* getKeyPointsConfidence();
  float getAverageKeyPointsConfidence();
  void enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant);
  unsigned findLargestBodyBox(NvAR_Rect& bodyBox, int variant = 0);
  unsigned acquireBodyBox(cv::Mat& src, NvAR_Rect& bodyBox, int variant = 0); 
  unsigned acquireBodyBoxAndKeyPoints(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Point3f* refKeyPoints3D,
      NvAR_Quaternion* refJointAngles, NvAR_BBoxes* refBodyBoxes, int variant = 0);
#if NV_MULTI_OBJECT_TRACKER
  unsigned acquireBodyBoxAndKeyPoints(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Point3f* refKeyPoints3D,
      NvAR_Quaternion* refJointAngles, NvAR_TrackingBBoxes* refBodyBoxes, int variant = 0);
#endif
  void setBodyStabilization(bool);
  void setMode(int);
  BodyEngine::Err setFocalLength(float);
  void useCudaGraph(bool); // Using cuda graph improves model latency
#if NV_MULTI_OBJECT_TRACKER
  void enablePeopleTracking(bool _bEnablePeopleTracking, unsigned int _shadowTrackingAge = 90, unsigned int _probationAge = 10, unsigned int _maxTargetsTracked = 30);
#endif
  int getNumKeyPoints() { return numKeyPoints; }
  std::vector<NvAR_Point3f> getReferencePose() { return referencePose; }

  NvCVImage inputImageBuffer{}, tmpImage{};
  NvAR_FeatureHandle bodyDetectHandle{}, keyPointDetectHandle{};
  std::vector<NvAR_Point2f> keypoints;
  std::vector<float> keypoints_confidence;
  std::vector<NvAR_Point3f> keypoints3D;
  std::vector<NvAR_Quaternion> jointAngles;
  CUstream stream{};
  std::vector<NvAR_Rect> output_bbox_data;
  std::vector<float> output_bbox_conf_data;
  NvAR_BBoxes output_bboxes{};
#if NV_MULTI_OBJECT_TRACKER
  NvAR_TrackingBBoxes output_tracking_bboxes{};
  std::vector<NvAR_TrackingBBox> output_tracking_bbox_data;
#endif
  
  int batchSize;
  int nvARMode;
  std::mt19937 ran;
  unsigned int numKeyPoints;
  std::vector<NvAR_Point3f> referencePose;
  float confidenceThreshold;
  std::string body_model;

  bool bStabilizeBody;
  bool bUseOTAU;
  char *bdOTAModelPath, *ldOTAModelPath;
  float bFocalLength;
  bool bUseCudaGraph;
#if NV_MULTI_OBJECT_TRACKER
  bool bEnablePeopleTracking;
  unsigned int shadowTrackingAge;
  unsigned int probationAge;
  unsigned int maxTargetsTracked;
#endif
  BodyEngine() {
    batchSize = 1;
    nvARMode = 1;
    bStabilizeBody = true;
    bUseCudaGraph = true;
#if NV_MULTI_OBJECT_TRACKER
    bEnablePeopleTracking = false;
    shadowTrackingAge = 90;
    probationAge = 10;
    maxTargetsTracked = 30;
#endif
    bFocalLength = FOCAL_LENGTH_DEFAULT;
    confidenceThreshold = 0.f;
    appMode = keyPointDetection;
    input_image_width = 960;
    input_image_height = 544;
    input_image_pitch = 3 * input_image_width * sizeof(unsigned char);  // RGB
    bUseOTAU = false;
    bdOTAModelPath = NULL;
    ldOTAModelPath = NULL;
  }
  enum mode { bodyDetection = 0, keyPointDetection } appMode;
  void setAppMode(BodyEngine::mode _mAppMode);
};
#endif
