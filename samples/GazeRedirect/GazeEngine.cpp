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
#include <iostream>
#include "GazeEngine.h"
#include "RenderingUtils.h"

bool CheckResult(NvCV_Status nvErr, unsigned line) {
  if (NVCV_SUCCESS == nvErr) return true;
  std::cout << "ERROR: " << NvCV_GetErrorStringFromCode(nvErr) << ", line " << line << std::endl;
  return false;
}

GazeEngine::Err GazeEngine::acquireGazeRedirection(cv::Mat& frame, cv::Mat& outputFrame) {
  GazeEngine::Err err = GazeEngine::Err::errNone;
  NvCV_Status nvErr;
  if (!frame.empty()) {
    NvCVImage fxSrcChunkyCPU;
    (void)NVWrapperForCVMat(&frame, &fxSrcChunkyCPU);
    nvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errGeneral);
  }
  nvErr = NvAR_Run(gazeRedirectHandle);

  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errRun);

  if (bGazeRedirect) {
    // Redirection is taking place. The feature has an output redirected image
    NvCVImage fxDstChunkyCPU;
    (void)NVWrapperForCVMat(&outputFrame, &fxDstChunkyCPU);
    nvErr = NvCVImage_Transfer(&outputImageBuffer, &fxDstChunkyCPU, 1.0f, stream, &tmpImage);
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errGeneral);
  } else {
    // Redirection is not taking place. There is no output image, therefore clone the input frame to output.
    outputFrame = frame.clone();
  }

  if (getAverageLandmarksConfidence() < confidenceThreshold) return GazeEngine::Err::errRun;
bail:
  if (err != Err::errNone) {
    printf("ERROR: An error has occured while running the Gaze Redirection \n");
  }
  return err;
}

NvAR_RenderingParams* GazeEngine::getRenderingParams() { return rendering_params; }

GazeEngine::Err GazeEngine::createGazeRedirectionFeature(const char* modelPath, unsigned int batchsize) {
  GazeEngine::Err err = GazeEngine::Err::errNone;
  NvCV_Status cuErr = NvAR_CudaStreamCreate(&stream);

  if (NVCV_SUCCESS != cuErr) {
    printf("Cannot create a cuda stream: %s\n", NvCV_GetErrorStringFromCode(cuErr));
    return errInitialization;
  }
  NvCV_Status nvErr;
  nvErr = NvAR_Create(NvAR_Feature_GazeRedirection, &gazeRedirectHandle);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errEffect);
  if (bUseOTAU && (!modelPath || !modelPath[0])) {
    nvErr = NvAR_SetString(gazeRedirectHandle, NvAR_Parameter_Config(ModelDir), this->fdOTAModelPath);
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  } else {
    nvErr = NvAR_SetString(gazeRedirectHandle, NvAR_Parameter_Config(ModelDir), modelPath);
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  }
  nvErr = NvAR_SetU32(gazeRedirectHandle, NvAR_Parameter_Config(Landmarks_Size), numLandmarks);  // TODO: Check if nonzero??
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  // Temporal flag set to -1 = 0xffffffff: turn on all filtering
  nvErr = NvAR_SetU32(gazeRedirectHandle, NvAR_Parameter_Config(Temporal), (bStabilizeFace ? -1 : 0));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  nvErr = NvAR_SetU32(gazeRedirectHandle, NvAR_Parameter_Config(GazeRedirect), bGazeRedirect);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  nvErr = NvAR_SetCudaStream(gazeRedirectHandle, NvAR_Parameter_Config(CUDAStream), stream);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  nvErr = NvAR_SetU32(gazeRedirectHandle, NvAR_Parameter_Config(UseCudaGraph), bUseCudaGraph);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  nvErr = NvAR_SetU32(gazeRedirectHandle, NvAR_Parameter_Config(EyeSizeSensitivity), eyeSizeSensitivity);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  nvErr = NvAR_Load(gazeRedirectHandle);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errInitialization);
bail:
  if (err != Err::errNone) {
    printf("ERROR: An error has occured while initializing Gaze Redirection\n");
  }
  return err;
}

GazeEngine::Err GazeEngine::initGazeRedirectionIOParams() {
  GazeEngine::Err err = GazeEngine::Err::errNone;
  unsigned int OUTPUT_SIZE_KPTS, OUTPUT_GAZE_SIZE = 2, HEAD_TRANSLATION_SIZE = 3;

  NvCV_Status nvErr = NvCVImage_Alloc(&inputImageBuffer, input_image_width, input_image_height, NVCV_BGR, NVCV_U8,
                                      NVCV_CHUNKY, NVCV_GPU, 1);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errInitialization);
  nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Input(Image), &inputImageBuffer, sizeof(NvCVImage));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  if (bGazeRedirect) {
    // Redirection is set. Allocate output image buffer.
    nvErr = NvCVImage_Alloc(&outputImageBuffer, input_image_width, input_image_height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY,
       NVCV_GPU, 1);
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errInitialization);
    nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(Image), &outputImageBuffer, sizeof(NvCVImage));
    BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);
  }
  nvErr = NvAR_SetS32(gazeRedirectHandle, NvAR_Parameter_Input(Width), input_image_width);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr = NvAR_SetS32(gazeRedirectHandle, NvAR_Parameter_Input(Height), input_image_height);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr = NvAR_GetU32(gazeRedirectHandle, NvAR_Parameter_Config(Landmarks_Size), &OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  facial_landmarks.assign(batchSize * OUTPUT_SIZE_KPTS, {0.f, 0.f});
  nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(Landmarks), facial_landmarks.data(),
                         sizeof(NvAR_Point2f));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  gaze_output_landmarks.assign(batchSize * num_output_landmarks , { 0.f, 0.f });
  nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(GazeOutputLandmarks), gaze_output_landmarks.data(),
    sizeof(NvAR_Point2f));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  facial_landmarks_confidence.assign(batchSize * OUTPUT_SIZE_KPTS, 0.f);
  nvErr = NvAR_SetF32Array(gazeRedirectHandle, NvAR_Parameter_Output(LandmarksConfidence),
                           facial_landmarks_confidence.data(), batchSize * OUTPUT_SIZE_KPTS);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(gazeRedirectHandle, NvAR_Parameter_Output(OutputGazeVector), gaze_angles_vector,
                           batchSize * OUTPUT_GAZE_SIZE);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr = NvAR_SetF32Array(gazeRedirectHandle, NvAR_Parameter_Output(OutputHeadTranslation), head_translation,
                           batchSize * HEAD_TRANSLATION_SIZE);
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(HeadPose), &head_pose, sizeof(NvAR_Quaternion));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  nvErr =
      NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(GazeDirection), &gaze_direction, sizeof(NvAR_Point3f));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

  output_bbox_data.assign(batchSize, {0.f, 0.f, 0.f, 0.f});
  output_bboxes.boxes = output_bbox_data.data();
  output_bboxes.max_boxes = (uint8_t)batchSize;
  output_bboxes.num_boxes = (uint8_t)batchSize;
  nvErr = NvAR_SetObject(gazeRedirectHandle, NvAR_Parameter_Output(BoundingBoxes), &output_bboxes, sizeof(NvAR_BBoxes));
  BAIL_IF_NVERR(nvErr, err, GazeEngine::Err::errParameter);

bail:
  if (err != Err::errNone) {
    printf("ERROR: An error has occured while setting input, output parmeters for Gaze Redirection\n");
  }
  return err;
}

void GazeEngine::destroyGazeRedirectionFeature() {
  if (stream) {
    NvAR_CudaStreamDestroy(stream);
    stream = 0;
  }
  output_bbox_data.clear();
  facial_landmarks.clear();
  facial_landmarks_confidence.clear();
  NvCVImage_Dealloc(&inputImageBuffer);
  NvCVImage_Dealloc(&outputImageBuffer);
  if (rendering_params) {
    delete rendering_params;
    rendering_params = nullptr;
  }
  if (gazeRedirectHandle) {
    (void)NvAR_Destroy(gazeRedirectHandle);
    gazeRedirectHandle = nullptr;
  }
}

unsigned GazeEngine::findFaceBoxes() {
  NvCV_Status nvErr = NvAR_Run(faceDetectHandle);
  if (NVCV_SUCCESS != nvErr) return 0;
  return (unsigned)output_bboxes.num_boxes;
}

NvAR_Rect* GazeEngine::getLargestBox() {
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

NvAR_BBoxes* GazeEngine::getBoundingBoxes() { return &output_bboxes; }

void GazeEngine::enlargeAndSquarifyImageBox(float enlarge, NvAR_Rect& box, int FLAG_variant) {
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

void GazeEngine::DrawPose(const cv::Mat& src, const NvAR_Quaternion* pose) const {
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

std::array<float, 2> GazeEngine::GetAverageLandmarkPositionInGlSpace() const{
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

void GazeEngine::DrawEstimatedGaze(const cv::Mat& src) {
  // Get the largest bounding box of the face.
  NvAR_Rect* pFaceBox = getLargestBox();

  std::vector<cv::Point3f> gaze_direction_3d;
  const float distance = 50;
  // Create opencv Point3f objects from direction points
  cv::Point3f gaze_direction_origin(gaze_direction[0].x, gaze_direction[0].y, gaze_direction[0].z);
  cv::Point3f gaze_direction_target;
  // Compute target point at a fixed distance from the gaze origin in the estimated gaze direction.
  gaze_direction_target.x = gaze_direction[0].x + distance * gaze_direction[1].x;
  gaze_direction_target.y = gaze_direction[0].y - distance * gaze_direction[1].y;
  gaze_direction_target.z = gaze_direction[0].z - distance * gaze_direction[1].z;

  // Compute 2D projections of the origin and target.
  gaze_direction_3d.push_back(gaze_direction_origin);
  gaze_direction_3d.push_back(gaze_direction_target);

  // Intialize camera matrix and coefficiennts for projecting into the image plane.
  float fx = (float)src.cols, fy = (float)src.cols, cx = (float)src.cols / 2.f, cy = (float) src.rows / 2.f;
  float camera_data[9] = {fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f};
  cv::Mat cam_matrix(3, 3, CV_32F, camera_data);
  cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);  // Distortion vector
  distCoeffs.at<double>(0) = 0;
  distCoeffs.at<double>(1) = 0;
  distCoeffs.at<double>(2) = 0;
  distCoeffs.at<double>(3) = 0;
  distCoeffs.at<double>(4) = 0;

  cv::Mat rVec(3, 1, cv::DataType<double>::type);  // Rotation vector
  rVec.at<double>(0) = 0;
  rVec.at<double>(1) = 0;
  rVec.at<double>(2) = 0;

  cv::Mat tVec(3, 1, cv::DataType<double>::type);  // Translation vector
  tVec.at<double>(0) = 0;
  tVec.at<double>(1) = 0;
  tVec.at<double>(2) = 0;

  std::vector<cv::Point2f> gaze_direction_2d;
  cv::projectPoints(gaze_direction_3d, rVec, tVec, cam_matrix, distCoeffs, gaze_direction_2d);

  // Plot the original and target points on the image plane.
  cv::line(src,
           cv::Point((int)(gaze_direction_2d[0].x),
                     (int)(gaze_direction_2d[0].y)),
           cv::Point((int)(gaze_direction_2d[1].x),
                     (int)(gaze_direction_2d[1].y)),
           cv::Scalar(0, 0, 255), 2);
}

NvCV_Status GazeEngine::findLandmarks() {
  NvCV_Status nvErr;

  nvErr = NvAR_Run(landmarkDetectHandle);
  if (NVCV_SUCCESS != nvErr) {
    return nvErr;
  }

  if (getAverageLandmarksConfidence() < confidenceThreshold) {
    return NVCV_ERR_GENERAL;
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
  return NVCV_SUCCESS;
}

NvAR_Point2f* GazeEngine::getLandmarks() { return facial_landmarks.data(); }

NvAR_Point2f* GazeEngine::getGazeOutputLandmarks() { return gaze_output_landmarks.data(); }

NvAR_Point3f* GazeEngine::getGazeDirectionPoints() { return gaze_direction; }

float* GazeEngine::getLandmarksConfidence() { return facial_landmarks_confidence.data(); }

float GazeEngine::getAverageLandmarksConfidence() {
  float average_confidence = 0.0f;
  float* keypoints_landmarks_confidence = getLandmarksConfidence();
  for (int i = 0; i < batchSize * numLandmarks; i++) {
    average_confidence += keypoints_landmarks_confidence[i];
  }
  average_confidence /= batchSize * numLandmarks;
  return average_confidence;
}

NvAR_Quaternion* GazeEngine::getPose() { return &head_pose; }
float* GazeEngine::getHeadTranslation() { return head_translation; }
float* GazeEngine::getGazeVector() { return gaze_angles_vector; }

unsigned GazeEngine::findLargestFaceBox(NvAR_Rect& faceBox, int variant) {
  unsigned n;
  NvAR_Rect* pFaceBox;

  n = findFaceBoxes();
  if (n >= 1) {
    pFaceBox = getLargestBox();
    if (nullptr == pFaceBox) {
      faceBox.x = faceBox.y = faceBox.width = faceBox.height = 0.0f;
    } else {
      faceBox = *pFaceBox;
    }
    enlargeAndSquarifyImageBox(.2f, faceBox, variant);
  }
  return n;
}

unsigned GazeEngine::acquireFaceBox(cv::Mat& src, NvAR_Rect& faceBox, int variant) {
  unsigned n = 0;
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return n;
  }

  n = findLargestFaceBox(faceBox, variant);
  return n;
}

unsigned GazeEngine::acquireFaceBoxAndLandmarks(cv::Mat& src, NvAR_Point2f* refMarks, NvAR_Rect& faceBox) {
  unsigned n = 0;
  NvCVImage fxSrcChunkyCPU;
  (void)NVWrapperForCVMat(&src, &fxSrcChunkyCPU);
  NvCV_Status cvErr = NvCVImage_Transfer(&fxSrcChunkyCPU, &inputImageBuffer, 1.0f, stream, &tmpImage);

  if (NVCV_SUCCESS != cvErr) {
    return n;
  }

  if (findLandmarks() != NVCV_SUCCESS) return 0;
  faceBox = output_bboxes.boxes[0];
  n = 1;
  memcpy(refMarks, getLandmarks(), sizeof(NvAR_Point2f) * numLandmarks);

  return n;
}

void GazeEngine::setFaceStabilization(bool _bStabilizeFace) { bStabilizeFace = _bStabilizeFace; }
void GazeEngine::setGazeRedirect(bool _bGazeRedirect) {
  // If this variable is set, gaze redirection occurs in addition to estimation.
  bGazeRedirect = _bGazeRedirect;
}

void GazeEngine::setUseCudaGraph(bool _bUseCudaGraph) {
  // If this variable is set, gaze redirection occurs in addition to estimation.
  bUseCudaGraph = _bUseCudaGraph;
}

GazeEngine::Err GazeEngine::setNumLandmarks(int n) {
  GazeEngine::Err err = errNone;
  for (auto const& info : LANDMARKS_INFO) {
    if (n == info.numPoints) {
      numLandmarks = info.numPoints;
      confidenceThreshold = info.confidence_threshold;
      return err;
    }
  }
  err = errGeneral;
  return err;
}

void GazeEngine::setEyeSizeSensitivity(unsigned _eyeSizeSensitivity) { eyeSizeSensitivity = _eyeSizeSensitivity; }