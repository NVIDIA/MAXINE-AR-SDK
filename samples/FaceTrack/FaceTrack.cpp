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
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "FaceEngine.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"
#include "RenderingUtils.h"

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
#define M_2PI 6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI ((float)M_PI)
#define F_PI_2 ((float)M_PI_2)
#define F_2PI ((float)M_2PI)

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */

#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

/********************************************************************************
 * Command-line arguments
 ********************************************************************************/

bool FLAG_debug = false, FLAG_verbose = false, FLAG_temporal = true, FLAG_captureOutputs = false,
     FLAG_offlineMode = false;
std::string FLAG_outDir, FLAG_inFile, FLAG_outFile, FLAG_modelPath, FLAG_landmarks, FLAG_proxyWireframe,
    FLAG_captureCodec = "avc1", FLAG_camRes;
unsigned int FLAG_batch = 1;

/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  printf(
      "AboutFace [<args> ...]\n"
      "where <args> is\n"
      " --verbose[=(true|false)]          report interesting info\n"
      " --debug[=(true|false)]            report debugging info\n"
      " --temporal[=(true|false)]         temporally optimize face rect and landmarks\n"
      " --capture_outputs[=(true|false)]  enables video/image capture and writing face detection/landmark outputs\n"
      " --offline_mode[=(true|false)]     disables webcam, reads video from file and writes output video results\n"
      " --cam_res=[WWWx]HHH               specify resolution as height or width x height\n"
      " --in_file=<file>                  specify the  input file\n"
      " --codec=<fourcc>                  FOURCC code for the desired codec (default H264)\n"
      " --in=<file>                       specify the  input file\n"
      " --out_file=<file>                 specify the output file\n"
      " --out=<file>                      specify the output file\n"
      " --model_path=<path>               specify the directory containing the TRT models\n"
      " --wireframe_mesh=<path>           specify the path to a proxy wireframe mesh\n"
      " --batch=<uint>                    1 - 8, used for batch inferencing in landmark detector "
      " --benchmarks[=<pattern>]          run benchmarks\n");
}

static bool GetFlagArgVal(const char *flag, const char *arg, const char **val) {
  if (*arg != '-') {
    return false;
  }
  while (*++arg == '-') {
    continue;
  }
  const char *s = strchr(arg, '=');
  if (s == NULL) {
    if (strcmp(flag, arg) != 0) {
      return false;
    }
    *val = NULL;
    return true;
  }
  unsigned n = (unsigned)(s - arg);
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0)) {
    return false;
  }
  *val = s + 1;
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, std::string *val) {
  const char *valStr;
  if (!GetFlagArgVal(flag, arg, &valStr)) return false;
  val->assign(valStr ? valStr : "");
  return true;
}

static bool GetFlagArgVal(const char *flag, const char *arg, bool *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = (valStr == NULL || strcasecmp(valStr, "true") == 0 || strcasecmp(valStr, "on") == 0 ||
            strcasecmp(valStr, "yes") == 0 || strcasecmp(valStr, "1") == 0);
  }
  return success;
}

bool GetFlagArgVal(const char *flag, const char *arg, long *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success) {
    *val = strtol(valStr, NULL, 10);
  }
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, unsigned *val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success) {
    *val = (unsigned)longVal;
  }
  return success;
}

/********************************************************************************
 * StringToFourcc
 ********************************************************************************/

static int StringToFourcc(const std::string &str) {
  union chint {
    int i;
    char c[4];
  };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;) x.c[n] = str[n];
  return x.i;
}

/********************************************************************************
 * ParseMyArgs
 ********************************************************************************/

static int ParseMyArgs(int argc, char **argv) {
  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char *arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&
               (GetFlagArgVal("verbose", arg, &FLAG_verbose) || GetFlagArgVal("debug", arg, &FLAG_debug) ||
                GetFlagArgVal("in", arg, &FLAG_inFile) || GetFlagArgVal("in_file", arg, &FLAG_inFile) ||
                GetFlagArgVal("out", arg, &FLAG_outFile) || GetFlagArgVal("out_file", arg, &FLAG_outFile) ||
                GetFlagArgVal("offline_mode", arg, &FLAG_offlineMode) ||
                GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||
                GetFlagArgVal("cam_res", arg, &FLAG_camRes) || GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||
                GetFlagArgVal("landmarks", arg, &FLAG_landmarks) || GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
                GetFlagArgVal("wireframe_mesh", arg, &FLAG_proxyWireframe) ||
                GetFlagArgVal("temporal", arg, &FLAG_temporal))) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      Usage();
    } else if (arg[1] != '-') {
      for (++arg; *arg; ++arg) {
        if (*arg == 'v') {
          FLAG_verbose = true;
        } else {
          // printf("Unknown flag: \"-%c\"\n", *arg);
        }
      }
      continue;
    } else {
      // printf("Unknown flag: \"%s\"\n", arg);
    }
  }
  return errs;
}

enum {
  myErrNone = 0,
  myErrShader = -1,
  myErrProgram = -2,
  myErrTexture = -3,
};

#if 1
class MyTimer {
 public:
  void start() { t0 = std::chrono::high_resolution_clock::now(); }       /**< Start  the timer. */
  void pause() { dt = std::chrono::high_resolution_clock::now() - t0; }  /**< Pause  the timer. */
  void resume() { t0 = std::chrono::high_resolution_clock::now() - dt; } /**< Resume the timer. */
  void stop() { pause(); }                                               /**< Stop   the timer. */
  double elapsedTimeFloat() const {
    return std::chrono::duration<double>(dt).count();
  } /**< Report the elapsed time as a float. */
 private:
  std::chrono::high_resolution_clock::time_point t0;
  std::chrono::high_resolution_clock::duration dt;
};
#endif

std::string getCalendarTime() {
  // Get the current time
  std::chrono::system_clock::time_point currentTimePoint = std::chrono::system_clock::now();
  // Convert to time_t from time_point
  std::time_t currentTime = std::chrono::system_clock::to_time_t(currentTimePoint);
  // Convert to tm to get structure holding a calendar date and time broken down into its components.
  std::tm brokenTime = *std::localtime(&currentTime);
  std::ostringstream calendarTime;
  calendarTime << std::put_time(
      &brokenTime,
      "%Y-%m-%d-%H-%M-%S");  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
  // Get the time since epoch 0(Thu Jan  1 00:00:00 1970) and the remainder after division is
  // our milliseconds
  std::chrono::milliseconds currentMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(currentTimePoint.time_since_epoch()) % 1000;
  // Append the milliseconds to the stream
  calendarTime << "-" << std::setfill('0') << std::setw(3) << currentMilliseconds.count();  // milliseconds
  return calendarTime.str();
}

class DoApp {
 public:
  enum Err {
    errNone,
    errUnimplemented,
    errMissing,
    errVideo,
    errImageSize,
    errNotFound,
    errFaceModelInit,
    errGLFWInit,
    errGLInit,
    errRendererInit,
    errGLResource,
    errGLGeneric,
    errFaceFit,
    errNoFace,
    errSDK,
    errCuda,
    errCancel,
    errInitFaceEngine
  };

  FaceEngine face_ar_engine;
  DoApp();
  ~DoApp();

  void stop();
  Err initFaceEngine(const char *modelPath = nullptr);
  Err initCamera(const char *camRes = nullptr);
  Err initOfflineMode(const char *inputFilename = nullptr, const char *outputFilename = nullptr);
  Err acquireFrame();
  Err acquireFaceBox();
  Err acquireFaceBoxAndLandmarks();
  Err fitFaceModel();
  Err run();
  void showFaceFitErrorMessage();
  void drawFPS(cv::Mat &img);
  void DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox);
  void DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks);
  void DrawFaceMesh(const cv::Mat &src, NvAR_FaceMesh *face_mesh);
  void drawKalmanStatus(cv::Mat &img);
  void drawVideoCaptureStatus(cv::Mat &img);
  Err setProxyWireframe(const char *path);
  void processKey(int key);
  void writeVideoAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks = NULL);
  void writeFrameAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks = NULL);
  void writeEstResults(std::ofstream &outputFile, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks = NULL);
  void getFPS();
  static const char *errorStringFromCode(Err code);

  cv::VideoCapture cap{};
  cv::Mat frame;
  int inputWidth, inputHeight;
  cv::VideoWriter faceDetectOutputVideo{}, landMarkOutputVideo{}, faceFittingOutputVideo{};
  int frameIndex;
  static const char windowTitle[];
  double frameTime;
  // std::chrono::high_resolution_clock::time_point frameTimer;
  MyTimer frameTimer;
  cv::VideoWriter capturedVideo;
  std::ofstream faceEngineVideoOutputFile;

  FaceEngine::Err nvErr;
  float expr[6];
  bool drawVisualization, showFPS, captureVideo, captureFrame;
  std::vector<std::array<uint16_t, 3>> proxyWireframe;
  float scaleOffsetXY[4];
  std::vector<NvAR_Vector3u16> wfMesh_tvi_data;
};

DoApp *gApp = nullptr;
const char DoApp::windowTitle[] = "WINDOW";

void DoApp::processKey(int key) {
  switch (key) {
    case '3':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::faceMeshGeneration);
      nvErr =  face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      // If there is an error, fallback to mode '2' i.e. landmark detection
      if (nvErr == FaceEngine::Err::errNone) {
        face_ar_engine.initFeatureIOParams();
        break;
      } else if (nvErr == FaceEngine::Err::errInitialization) {
        showFaceFitErrorMessage();
      }
    case '2':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::landmarkDetection);
      face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      face_ar_engine.initFeatureIOParams();
      break;
    case '1':
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::faceDetection);
      face_ar_engine.createFeatures(FLAG_modelPath.c_str());
      face_ar_engine.initFeatureIOParams();
      break;
    case 'C':
    case 'c':
      captureVideo = !captureVideo;
      break;
    case 'S':
    case 's':
      captureFrame = !captureFrame;
      break;
    case 'W':
    case 'w':
      drawVisualization = !drawVisualization;
      break;
    case 'F':
    case 'f':
      showFPS = !showFPS;
      break;
    default:
      break;
  }
}

DoApp::Err DoApp::initFaceEngine(const char *modelPath) {
  Err err = errNone;

  if (!cap.isOpened()) return errVideo;

  nvErr = face_ar_engine.createFeatures(modelPath);
  if (nvErr != FaceEngine::Err::errNone) {
    if (nvErr == FaceEngine::Err::errInitialization && face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration) {
      showFaceFitErrorMessage();
      face_ar_engine.destroyFeatures();
      face_ar_engine.setAppMode(FaceEngine::mode::landmarkDetection);
      nvErr = face_ar_engine.createFeatures(modelPath);
    }
    if (nvErr != FaceEngine::Err::errNone)
      err = errInitFaceEngine;
  }

#ifdef DEBUG
  detector->setOutputLocation(outputDir);
#endif  // DEBUG

#define VISUALIZE
#ifdef VISUALIZE
  if (!FLAG_offlineMode) cv::namedWindow(windowTitle, 1);
#endif  // VISUALIZE

  frameIndex = 0;

  return err;
}

void DoApp::stop() {
  face_ar_engine.destroyFeatures();

  if (FLAG_offlineMode) {
    faceDetectOutputVideo.release();
    landMarkOutputVideo.release();
    faceFittingOutputVideo.release();
  }
  cap.release();
#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}

void DoApp::showFaceFitErrorMessage() {
  cv::Mat errBox = cv::Mat::zeros(120, 640, CV_8UC3);
  cv::putText(errBox, cv::String("Warning: Face Fitting needs face_model0.nvf in the path --model_path"), cv::Point(20, 20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(errBox, cv::String("or in NVAR_MODEL_DIR environment variable."), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(errBox, cv::String("See samples\\FaceTrack\\Readme.txt"), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(255, 255, 255), 1);
  cv::putText(errBox, cv::String("Press any key to continue with other features"), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(255, 255, 255), 1);
  cv::imshow(windowTitle, errBox);
  cv::waitKey(0);
}

void DoApp::DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox) {
  cv::Mat frame;
  if (FLAG_offlineMode)
    frame = src.clone();
  else
    frame = src;

  if (output_bbox)
    cv::rectangle(frame, cv::Point((int)output_bbox->x, (int)output_bbox->y),
                  cv::Point((int)output_bbox->x + output_bbox->width, (int)output_bbox->y + output_bbox->height),
                  cv::Scalar(255, 0, 0), 2);
  if (FLAG_offlineMode) faceDetectOutputVideo.write(frame);
}

void DoApp::writeVideoAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks) {
  if (captureVideo) {
    if (!capturedVideo.isOpened()) {
      const std::string currentCalendarTime = getCalendarTime();
      const std::string capturedOutputFileName = currentCalendarTime + ".mp4";
      getFPS();
      if (frameTime) {
        float fps = 1. / frameTime;
        capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps,
                           cv::Size(frame.cols, frame.rows));
        if (!capturedVideo.isOpened()) {
          std::cout << "Error: Could not open video: \"" << capturedOutputFileName << "\"\n";
          return;
        }
        if (FLAG_verbose) {
          std::cout << "Capturing video started" << std::endl;
        }
      } else {  // If frameTime is 0.f, returns without writing the frame to the Video
        return;
      }
      const std::string outputsFileName = currentCalendarTime + ".txt";
      faceEngineVideoOutputFile.open(outputsFileName, std::ios_base::out);
      if (!faceEngineVideoOutputFile.is_open()) {
        std::cout << "Error: Could not open file: \"" << outputsFileName << "\"\n";
        return;
      }
      std::string landmarkDetectionMode = (landmarks == NULL) ? "Off" : "On";
      faceEngineVideoOutputFile << "// FaceDetectOn, LandmarkDetect" << landmarkDetectionMode << "\n ";
      faceEngineVideoOutputFile
          << "// kNumFaces, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumFaces}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
    }
    // Write each frame to the Video
    capturedVideo << frame;
    writeEstResults(faceEngineVideoOutputFile, output_bboxes, landmarks);
  } else {
    if (capturedVideo.isOpened()) {
      if (FLAG_verbose) {
        std::cout << "Capturing video ended" << std::endl;
      }
      capturedVideo.release();
      if (faceEngineVideoOutputFile.is_open()) faceEngineVideoOutputFile.close();
    }
  }
}

void DoApp::writeEstResults(std::ofstream &outputFile, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks) {
  /**
   * Output File Format :
   * FaceDetectOn, LandmarkDetectOn
   * kNumFaces, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumFaces}, kNumLMs, [lm_x, lm_y]{kNumLMs}
   */

  int faceDetectOn = (face_ar_engine.appMode == FaceEngine::mode::faceDetection ||
                      face_ar_engine.appMode == FaceEngine::mode::landmarkDetection ||
                      face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration)
                         ? 1
                         : 0;
  int landmarkDetectOn = (face_ar_engine.appMode == FaceEngine::mode::landmarkDetection ||
                          face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration)
                             ? 1
                             : 0;
  outputFile << faceDetectOn << "," << landmarkDetectOn << "\n";

  if (faceDetectOn && output_bboxes.num_boxes) {
    // Append number of faces detected in the current frame
    outputFile << unsigned(output_bboxes.num_boxes) << ",";
    // write outputbboxes to outputFile
    for (size_t i = 0; i < output_bboxes.num_boxes; i++) {
      int x1 = (int)output_bboxes.boxes[i].x, y1 = (int)output_bboxes.boxes[i].y,
          width = (int)output_bboxes.boxes[i].width, height = (int)output_bboxes.boxes[i].height;
      outputFile << x1 << "," << y1 << "," << width << "," << height << ",";
    }
  } else {
    outputFile << "0,";
  }
  if (landmarkDetectOn && output_bboxes.num_boxes) {
    // Append number of landmarks
    outputFile << FaceEngine::NUM_LANDMARKS << ",";
    // Append NUM_LANDMARKS * 2 points
    NvAR_Point2f *pt, *endPt;
    for (endPt = (pt = (NvAR_Point2f *)landmarks) + FaceEngine::NUM_LANDMARKS; pt < endPt; ++pt)
      outputFile << pt->x << "," << pt->y << ",";
  } else {
    outputFile << "0,";
  }

  outputFile << "\n";
}

void DoApp::writeFrameAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *landmarks) {
  if (captureFrame) {
    const std::string currentCalendarTime = getCalendarTime();
    const std::string capturedFrame = currentCalendarTime + ".png";
    cv::imwrite(capturedFrame, frame);
    if (FLAG_verbose) {
      std::cout << "Captured the frame" << std::endl;
    }
    // Write Face Engine Outputs
    const std::string outputFilename = currentCalendarTime + ".txt";
    std::ofstream outputFile;
    outputFile.open(outputFilename, std::ios_base::out);
    if (!outputFile.is_open()) {
      std::cout << "Error: Could not open file: \"" << outputFilename << "\"\n";
      return;
    }
    std::string landmarkDetectionMode = (landmarks == NULL) ? "Off" : "On";
    outputFile << "// FaceDetectOn, LandmarkDetect" << landmarkDetectionMode << "\n";
    outputFile << "// kNumFaces, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumFaces}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
    writeEstResults(outputFile, output_bboxes, landmarks);
    if (outputFile.is_open()) outputFile.close();
    captureFrame = false;
  }
}

void DoApp::DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks) {
  cv::Mat frame;
  if (FLAG_offlineMode)
    frame = src.clone();
  else
    frame = src;
  NvAR_Point2f *pt, *endPt;
  for (endPt = (pt = (NvAR_Point2f *)facial_landmarks) + FaceEngine::NUM_LANDMARKS; pt < endPt; ++pt)
    cv::circle(frame, cv::Point(lround(pt->x), lround(pt->y)), 1, cv::Scalar(0, 0, 255), -1);
  NvAR_Quaternion *pose = face_ar_engine.getPose();
  if (pose)
    face_ar_engine.DrawPose(frame, pose);
  if (FLAG_offlineMode) landMarkOutputVideo.write(frame);
}

void DoApp::DrawFaceMesh(const cv::Mat &src, NvAR_FaceMesh *face_mesh) {
  cv::Mat render_frame;
  if (FLAG_offlineMode)
    render_frame = src.clone();
  else
    render_frame = src;

  const cv::Scalar wfColor(0, 160, 0, 255);
  //Low resolution mesh can be used for visualization only
  if (proxyWireframe.size() > 0) {
    NvAR_FaceMesh wfMesh{nullptr, 0, nullptr, 0};
    wfMesh.num_vertices = face_mesh->num_vertices;
    wfMesh.vertices = face_mesh->vertices;
    wfMesh.num_tri_idx = proxyWireframe.size();
    wfMesh_tvi_data.resize(wfMesh.num_tri_idx);
    wfMesh.tvi = wfMesh_tvi_data.data();

    for (int i = 0; i < proxyWireframe.size(); i++) {
      auto proxyWireframe_idx = proxyWireframe[i];
      wfMesh.tvi[i].vec[0] = proxyWireframe_idx[0];
      wfMesh.tvi[i].vec[1] = proxyWireframe_idx[1];
      wfMesh.tvi[i].vec[2] = proxyWireframe_idx[2];
    }
    draw_wireframe(render_frame, wfMesh, *face_ar_engine.getRenderingParams(), wfColor);
  } else {
    draw_wireframe(render_frame, *face_mesh, * face_ar_engine.getRenderingParams(), wfColor);
  }

  if (FLAG_offlineMode) faceFittingOutputVideo.write(render_frame);
}

DoApp::Err DoApp::acquireFrame() {
  Err err = errNone;

  // If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
  // frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
  // resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
  // done here) as well as reallocate memory accordingly with FaceEngine::initFeatureIOParams()
  cap >> frame;  // get a new frame from camera
  if (frame.empty()) {
    // if in Offline mode, this means end of video,so we return
    if (FLAG_offlineMode) return errVideo;
    // try Init one more time if reading frames from camera
    initCamera(FLAG_camRes.c_str());
    cap >> frame;
    if (frame.empty()) return errVideo;
  }

  return err;
}

DoApp::Err DoApp::acquireFaceBox() {
  Err err = errNone;
  NvAR_Rect output_bbox;

  // get landmarks in  original image resolution coordinate space
  unsigned n = face_ar_engine.acquireFaceBox(frame, output_bbox, 0);

  if (n && FLAG_verbose) {
    printf("FaceBox: [\n");
    printf("%7.1f%7.1f%7.1f%7.1f\n", output_bbox.x, output_bbox.y, output_bbox.x + output_bbox.width,
           output_bbox.y + output_bbox.height);
    printf("]\n");
  }
  if (FLAG_captureOutputs) {
    writeFrameAndEstResults(frame, face_ar_engine.output_bboxes);
    writeVideoAndEstResults(frame, face_ar_engine.output_bboxes);
  }
  if (0 == n) return errNoFace;

#ifdef VISUALIZE

  if (drawVisualization) {
    DrawBBoxes(frame, &output_bbox);
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::acquireFaceBoxAndLandmarks() {
  Err err = errNone;

  NvAR_Rect output_bbox;
  NvAR_Point2f facial_landmarks[FaceEngine::NUM_LANDMARKS];

  // get landmarks in  original image resolution coordinate space
  unsigned n = face_ar_engine.acquireFaceBoxAndLandmarks(frame, facial_landmarks, output_bbox, 0);

  if (n && FLAG_verbose && face_ar_engine.appMode != FaceEngine::mode::faceDetection) {
    printf("Landmarks: [\n");
    NvAR_Point2f *pt, *endPt;
    for (endPt = (pt = (NvAR_Point2f *)facial_landmarks) + FaceEngine::NUM_LANDMARKS; pt < endPt; ++pt)
      printf("%7.1f%7.1f\n", pt->x, pt->y);
    printf("]\n");
  }
  if (FLAG_captureOutputs) {
    writeFrameAndEstResults(frame, face_ar_engine.output_bboxes, facial_landmarks);
    writeVideoAndEstResults(frame, face_ar_engine.output_bboxes, facial_landmarks);
  }
  if (0 == n) return errNoFace;

#ifdef VISUALIZE

  if (drawVisualization) {
    DrawLandmarkPoints(frame, facial_landmarks);
    if (FLAG_offlineMode) {
      DrawBBoxes(frame, &output_bbox);
    }
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::initCamera(const char *camRes) {
  if (cap.open(0)) {
    if (camRes) {
      int n;
      n = sscanf(camRes, "%d%*[xX]%d", &inputWidth, &inputHeight);
      switch (n) {
        case 2:
          break;  // We have read both width and height
        case 1:
          inputHeight = inputWidth;
          inputWidth = (int)(inputHeight * (4. / 3.) + .5);
          break;
        default:
          inputHeight = 0;
          inputWidth = 0;
          break;
      }
      if (inputWidth) cap.set(CV_CAP_PROP_FRAME_WIDTH, inputWidth);
      if (inputHeight) cap.set(CV_CAP_PROP_FRAME_HEIGHT, inputHeight);

      inputWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
      inputHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
      face_ar_engine.setInputImageWidth(inputWidth);
      face_ar_engine.setInputImageHeight(inputHeight);
    }
  } else
    return errVideo;
  return errNone;
}

DoApp::Err DoApp::initOfflineMode(const char *inputFilename, const char *outputFilename) {
  if (cap.open(inputFilename)) {
    inputWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    inputHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    face_ar_engine.setInputImageWidth(inputWidth);
    face_ar_engine.setInputImageHeight(inputHeight);
  } else {
    return Err::errNotFound;
  }

  std::string fdOutputVideoName, fldOutputVideoName, ffOutputVideoName;
  std::string outputFilePrefix;
  if (outputFilename && strlen(outputFilename) != 0) {
    outputFilePrefix = outputFilename;
  } else {
    size_t lastindex = std::string(inputFilename).find_last_of(".");
    outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
  }
  fdOutputVideoName = outputFilePrefix + "_bbox.mp4";
  fldOutputVideoName = outputFilePrefix + "_landmarks.mp4";
  ffOutputVideoName = outputFilePrefix + "_faceModel.mp4";

  if (!faceDetectOutputVideo.open(fdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                  cv::Size(inputWidth, inputHeight)))
    return Err::errSDK;
  if (!landMarkOutputVideo.open(fldOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                cv::Size(inputWidth, inputHeight)))
    return Err::errSDK;
  if (!faceFittingOutputVideo.open(ffOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                   cv::Size(inputWidth, inputHeight)))
    return Err::errSDK;

  return Err::errNone;
}

DoApp::Err DoApp::fitFaceModel() {
  DoApp::Err doErr = errNone;
  nvErr = face_ar_engine.fitFaceModel(frame);
  if (FLAG_captureOutputs) {
    writeFrameAndEstResults(frame, face_ar_engine.output_bboxes, face_ar_engine.getLandmarks());
    writeVideoAndEstResults(frame, face_ar_engine.output_bboxes, face_ar_engine.getLandmarks());
  }

  if (FaceEngine::Err::errNone == nvErr) {
#ifdef VISUALIZE
    frameTimer.pause();
    if (drawVisualization) {
      DrawFaceMesh(frame, face_ar_engine.getFaceMesh());
      if (FLAG_offlineMode) {
        DrawLandmarkPoints(frame, face_ar_engine.getLandmarks());
        DrawBBoxes(frame, face_ar_engine.getLargestBox());
      }
    }
    frameTimer.resume();
#endif  // VISUALIZE
  } else {
    doErr = errFaceFit;
  }

  return doErr;
}

DoApp::DoApp() {
  // Make sure things are initialized properly
  gApp = this;
  drawVisualization = true;
  showFPS = false;
  captureVideo = false;
  captureFrame = false;
  frameTime = 0;
  frameIndex = 0;
  nvErr = FaceEngine::errNone;
  scaleOffsetXY[0] = scaleOffsetXY[2] = 1.f;
  scaleOffsetXY[1] = scaleOffsetXY[3] = 0.f;
}

DoApp::~DoApp() {}

void DoApp::getFPS() {
  const float timeConstant = 16.f;
  frameTimer.stop();
  float t = (float)frameTimer.elapsedTimeFloat();
  if (t < 100.f) {
    if (frameTime)
      frameTime += (t - frameTime) * (1.f / timeConstant);  // 1 pole IIR filter
    else
      frameTime = t;
  } else {            // Ludicrous time interval; reset
    frameTime = 0.f;  // WAKE UP
  }
  frameTimer.start();
}

void DoApp::drawFPS(cv::Mat &img) {
  getFPS();
  if (frameTime && showFPS) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", 1. / frameTime);
    cv::putText(img, buf, cv::Point(img.cols - 80, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 255, 255), 1);
  }
}

void DoApp::drawKalmanStatus(cv::Mat &img) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Kalman %s", (face_ar_engine.bStabilizeFace ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

void DoApp::drawVideoCaptureStatus(cv::Mat &img) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Video Capturing %s", (captureVideo ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

DoApp::Err DoApp::run() {
  DoApp::Err doErr = errNone;

  face_ar_engine.initFeatureIOParams();

  while (1) {
    doErr = acquireFrame();
    if (doErr != DoApp::errNone) return doErr;
    if (face_ar_engine.appMode == FaceEngine::mode::faceDetection) {
      doErr = acquireFaceBox();
    } else if (face_ar_engine.appMode == FaceEngine::mode::landmarkDetection) {
      doErr = acquireFaceBoxAndLandmarks();
    } else if (face_ar_engine.appMode == FaceEngine::mode::faceMeshGeneration) {
      doErr = fitFaceModel();
    }
    if ((DoApp::errNoFace == doErr || DoApp::errFaceFit == doErr) && FLAG_offlineMode) {
      faceDetectOutputVideo.write(frame);
      landMarkOutputVideo.write(frame);
      faceFittingOutputVideo.write(frame);
    }
    if (DoApp::errCancel == doErr || DoApp::errVideo == doErr) return doErr;
    if (!frame.empty() && !FLAG_offlineMode) {
      if (drawVisualization) {
        drawFPS(frame);
        drawKalmanStatus(frame);
        if (FLAG_captureOutputs && captureVideo) drawVideoCaptureStatus(frame);
      }
      cv::imshow(windowTitle, frame);
    }

    if (!FLAG_offlineMode) {
      int n = cv::waitKey(1);
      if (n >= 0) {
        static const int ESC_KEY = 27;
        if (n == ESC_KEY) break;
        processKey(n);
      }
    }
  }
  return doErr;
}

DoApp::Err DoApp::setProxyWireframe(const char *path) {
  std::ifstream file(path);
  if (!file.is_open()) return errFaceModelInit;
  std::string lineBuf;
  while (getline(file, lineBuf)) {
    std::array<uint16_t, 3> tri;
    if (3 == sscanf(lineBuf.c_str(), "f %hd %hd %hd", &tri[0], &tri[1], &tri[2])) {
      --tri[0]; /* Convert from 1-based to 0-based indices */
      --tri[1];
      --tri[2];
      proxyWireframe.push_back(tri);
    }
  }
  file.close();
  return errNone;
}

const char *DoApp::errorStringFromCode(DoApp::Err code) {
  struct LUTEntry {
    Err code;
    const char *str;
  };
  static const LUTEntry lut[] = {
      {errNone, "no error"},
      {errUnimplemented, "the feature is unimplemented"},
      {errMissing, "missing input parameter"},
      {errVideo, "no video source has been found"},
      {errImageSize, "the image size cannot be accommodated"},
      {errNotFound, "the item cannot be found"},
      {errFaceModelInit, "face model initialization failed"},
      {errGLFWInit, "GLFW initialization failed"},
      {errGLInit, "OpenGL initialization failed"},
      {errRendererInit, "renderer initialization failed"},
      {errGLResource, "an OpenGL resource could not be found"},
      {errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
      {errFaceFit, "an error has occurred while face fitting"},
      {errNoFace, "no face has been found"},
      {errSDK, "an SDK error has occurred"},
      {errCuda, "a CUDA error has occurred"},
      {errCancel, "the user cancelled"},
      {errInitFaceEngine, "an error occurred while initializing the Face Engine"},
  };
  for (const LUTEntry *p = lut; p < &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
    if (p->code == code) return p->str;
  static char msg[18];
  snprintf(msg, sizeof(msg), "error #%d", code);
  return msg;
}

/********************************************************************************
 * main
 ********************************************************************************/

int main(int argc, char **argv) {
  DoApp app;
  DoApp::Err doErr;
  NvCV_Status nvErr;

  // Parse the arguments
  if (0 != ParseMyArgs(argc, argv)) return -100;

  if (FLAG_verbose) printf("Enable temporal optimizations in detecting face and landmarks = %d\n", FLAG_temporal);
  app.face_ar_engine.setFaceStabilization(FLAG_temporal);

  doErr = DoApp::errFaceModelInit;
  if (FLAG_modelPath.empty()) {
    printf("WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/face/models\n"
      "SDK will attempt to load the models from NVAR_MODEL_DIR environment variable");
  }

  if (FLAG_offlineMode) {
    if (FLAG_inFile.empty()) {
      doErr = DoApp::errMissing;
      printf("ERROR: %s, please specify input file using --in_file or --in \n", app.errorStringFromCode(doErr));
      goto bail;
    }
    app.initOfflineMode(FLAG_inFile.c_str(), FLAG_outFile.c_str());
  } else {
    app.initCamera(FLAG_camRes.c_str());
  }
  doErr = app.initFaceEngine(FLAG_modelPath.c_str());
  if (DoApp::errNone != doErr) {
    printf("ERROR: %s\n", app.errorStringFromCode(doErr));
    goto bail;
  }

  if (!FLAG_proxyWireframe.empty()) app.setProxyWireframe(FLAG_proxyWireframe.c_str());

  doErr = app.run();

bail:
  app.stop();
  return (int)doErr;
}
