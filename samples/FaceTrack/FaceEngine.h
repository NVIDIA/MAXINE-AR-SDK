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
#ifndef __FACE_ENGINE__
#define __FACE_ENGINE__

#include <random>
#include "nvAR.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"
#include "FeatureVertexName.h"
#define FITFACE_PRIVATE

class KalmanFilter1D {
 private:
  float Q_;          // Covariance of the process noise
  float xhat_;       // Current prediction
  float xhatminus_;  // Previous prediction
  float P_;          // Estimated accuracy of xhat_
  float Pminus_;     // Previous P_
  float K_;          // Kalman gain
  float R_;          // Covariance of the observation noise
  bool bFirstUse;

 public:
  KalmanFilter1D() { reset(); }

  KalmanFilter1D(float Q, float R) { reset(Q, R); }

  void reset() {
    R_ = 0.005f * 0.005f;
    Q_ = 1e-5f;
    xhat_ = 0.0f;
    xhatminus_ = 0.0f;
    P_ = 1;
    bFirstUse = true;
    Pminus_ = 0.0f;
    K_ = 0.0f;
  }

  void reset(float Q, float R) {
    reset();
    Q_ = Q;
    R_ = R;
  }

  float update(float val) {
    if (bFirstUse) {
      xhat_ = val;
      bFirstUse = false;
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

#define BAIL_IF_NVERR(nvErr, err, code)  \
  do {                                   \
    if (!CheckResult(nvErr, __LINE__)) { \
      err = code;                        \
      goto bail;                         \
    }                                    \
  } while (0)

typedef struct LandmarksProperties {
  int numPoints;
  float confidence_threshold;
}LandmarksProperties;

/********************************************************************************
 * FaceEngine
 ********************************************************************************/

class FaceEngine {
 public:
  enum Err { errNone, errGeneral, errRun, errInitialization, errRead, errEffect, errParameter, errNoFaceDetected };
  int input_image_width, input_image_height, input_image_pitch;
  const LandmarksProperties LANDMARKS_INFO[2] = {
           { 68,  15.0f }, // number of landmark points, confidence threshold value
           { 126, 15.0f}, // 
  };

  // Keep Older Confidence Threshold for Linux Models.
  const LandmarksProperties LANDMARKS_INFO_UNIX[2] = {
           { 68,  10.0f }, // number of landmark points, confidence threshold value
           { 126, 5.0f}, // 
  };

  void setInputImageWidth(int width) { input_image_width = width; }
  void setInputImageHeight(int height) { input_image_height = height; }
  int getInputImageWidth() { return input_image_width; }
  int getInputImageHeight() { return input_image_height; }
  int getInputImagePitch() { return input_image_pitch = input_image_width * 3 * sizeof(unsigned char); }
  void setFaceModel(const char *faceModel) { face_model = faceModel; }

  Err createFeatures(const char* modelPath, unsigned int _batchSize = 1, unsigned int mode = 0);
  Err createFaceDetectionFeature(const char* modelPath, CUstream stream);
  Err createLandmarkDetectionFeature(const char* modelPath, unsigned int batchSize, CUstream stream, unsigned int mode = 0);
  Err createFaceFittingFeature(const char* modelPath, CUstream stream);
  void destroyFeatures();
  void destroyFaceDetectionFeature();
  void destroyLandmarkDetectionFeature();
  void destroyFaceFittingFeature();
  Err initFeatureIOParams();
  Err initFaceDetectionIOParams(NvCVImage* _inputImageBuffer);
  Err initLandmarkDetectionIOParams(NvCVImage* _inputImageBuffer);
  Err initFaceFittingIOParams(NvCVImage* _inputImageBuffer);
  void releaseFeatureIOParams();
  void releaseFaceDetectionIOParams();
  void releaseLandmarkDetectionIOParams();
  void releaseFaceFittingIOParams();

  NvCV_Status findFaceBoxes(unsigned &num_boxes);
  NvAR_Rect* getLargestBox();
  FaceEngine::Err findLandmarks();
  NvAR_BBoxes* getBoundingBoxes();
  NvAR_Point2f* getLandmarks();
  NvAR_Quaternion* getPose();
  float* getLandmarksConfidence();
  float getAverageLandmarksConfidence();
  void enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant);
  static void jiggleBox(std::mt19937& ran, float minMag, float maxMag, const NvAR_Rect& cleanBox, NvAR_Rect& noisyBox);
  FaceEngine::Err findLargestFaceBox(NvAR_Rect& faceBox, int variant = 0);
  FaceEngine::Err acquireFaceBox(cv::Mat& src, NvAR_Rect& faceBox,int variant = 0); 
  FaceEngine::Err acquireFaceBoxAndLandmarks(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Rect& faceBox, int variant = 0);
  Err fitFaceModel(cv::Mat& frame);
  NvAR_FaceMesh* getFaceMesh();
  NvAR_RenderingParams* getRenderingParams();
  float* getShapeEigenvalues();
  float* getExpressionCoefficients();
  void setFaceStabilization(bool);
  int getNumShapeEigenvalues();
  int getNumExpressionCoefficients();
  Err setNumLandmarks(int);
  int getNumLandmarks() { return numLandmarks; }
  void DrawPose(const cv::Mat& src, const NvAR_Quaternion* pose) const;
  std::array<float, 2> GetAverageLandmarkPositionInGlSpace() const;

  NvCVImage inputImageBuffer{}, tmpImage{}, outputImageBuffer{};
  NvAR_FeatureHandle faceDetectHandle{}, landmarkDetectHandle{}, faceFitHandle{};
  std::vector<NvAR_Point2f> facial_landmarks;
  std::vector<float> facial_landmarks_confidence;
  std::vector<NvAR_Quaternion> facial_pose;
  NvAR_FaceMesh* face_mesh{};
  std::vector<NvAR_Vector3f> m_vertices;
  std::vector<NvAR_Vector3u16> m_triangles;
  NvAR_RenderingParams* rendering_params{};
  std::vector<float> shapeEigenvalues, expressionCoefficients;
  CUstream stream{};
  std::vector<NvAR_Rect> output_bbox_data;
  std::vector<float> output_bbox_conf_data;
  NvAR_BBoxes output_bboxes{};
  int batchSize;
  std::mt19937 ran;
  int numLandmarks;
  float confidenceThreshold;
  std::string face_model;
  
  bool bStabilizeFace;
  bool bUseOTAU;
  char *fdOTAModelPath, *ldOTAModelPath;

  FaceEngine() {
    batchSize = 1;
    bStabilizeFace = true;
    numLandmarks = LANDMARKS_INFO[0].numPoints;
    confidenceThreshold = LANDMARKS_INFO[0].confidence_threshold;
    appMode = faceMeshGeneration;
    input_image_width = 640;
    input_image_height = 480;
    input_image_pitch = 3 * input_image_width * sizeof(unsigned char);  // RGB
    bUseOTAU = false;
    fdOTAModelPath = NULL;
    ldOTAModelPath = NULL;
  }
  enum mode { faceDetection = 0, landmarkDetection, faceMeshGeneration
  } appMode;
  void setAppMode(FaceEngine::mode _mAppMode);
};
#endif
