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
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "BodyEngine.h"
#include "RenderingUtils.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#endif

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

#define DEBUG_RUNTIME

#define PEOPLE_TRACKING_BATCH_SIZE 8

/********************************************************************************
 * Command-line arguments
 ********************************************************************************/

bool FLAG_debug = false, FLAG_verbose = false, FLAG_temporal = true, FLAG_captureOutputs = false,
     FLAG_offlineMode = false, FLAG_useCudaGraph = true;
std::string FLAG_outDir, FLAG_inFile, FLAG_outFile, FLAG_modelPath, FLAG_captureCodec = "avc1",
            FLAG_camRes, FLAG_bodyModel;
unsigned int FLAG_appMode = 1, FLAG_mode = 1, FLAG_camindex=0;
#if NV_MULTI_OBJECT_TRACKER
bool FLAG_enablePeopleTracking = false;
unsigned int FLAG_shadowTrackingAge = 90, FLAG_probationAge = 10, FLAG_maxTargetsTracked = 30;
#endif
/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  printf(
      "BodyTrack [<args> ...]\n"
      "where <args> is\n"
      " --verbose[=(true|false)]          report interesting info\n"
      " --debug[=(true|false)]            report debugging info\n"
      " --temporal[=(true|false)]         temporally optimize body rect and keypoints\n"
      " --use_cuda_graph[=(true|false)]   enable faster execution by using cuda graph to capture engine execution\n"
      " --capture_outputs[=(true|false)]  enables video/image capture and writing body detection/keypoints outputs\n"
      " --offline_mode[=(true|false)]     disables webcam, reads video from file and writes output video results\n"
      " --cam_res=[WWWx]HHH               specify resolution as height or width x height\n"
      " --in_file=<file>                  specify the  input file\n"
      " --codec=<fourcc>                  FOURCC code for the desired codec (default H264)\n"
      " --in=<file>                       specify the  input file\n"
      " --out_file=<file>                 specify the output file\n"
      " --out=<file>                      specify the output file\n"
      " --model_path=<path>               specify the directory containing the TRT models\n"
      " --mode[=0|1]                      Model Mode. 0: High Quality, 1: High Performance\n"
      " --app_mode[=(0|1)]                App mode. 0: Body detection, 1: Keypoint detection "
#if NV_MULTI_OBJECT_TRACKER
      " --enable_people_tracking[=(0|1)]  Enables people tracking "
      " --shadow_tracking_age             Shadow Tracking Age after which tracking information of a person is removed. Measured in frames"
      " --probation_age                   Length of probationary period. Measured in frames"
      " --max_targets_tracked             Maximum number of targets to be tracked "
#endif
      "(Default).\n"
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
                GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
                GetFlagArgVal("app_mode", arg, &FLAG_appMode) ||
                GetFlagArgVal("mode", arg, &FLAG_mode) ||
                GetFlagArgVal("camindex", arg, &FLAG_camindex) ||
                GetFlagArgVal("use_cuda_graph", arg, &FLAG_useCudaGraph) ||
#if NV_MULTI_OBJECT_TRACKER
                GetFlagArgVal("enable_people_tracking", arg, &FLAG_enablePeopleTracking) ||
                GetFlagArgVal("shadow_tracking_age", arg, &FLAG_shadowTrackingAge) ||
                GetFlagArgVal("probation_age", arg, &FLAG_probationAge) ||
                GetFlagArgVal("max_targets_tracked", arg, &FLAG_maxTargetsTracked) ||
#endif
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

static const cv::Scalar cv_colors[] = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) };
enum {
  kColorRed = 0,
  kColorGreen = 1,
  kColorBlue = 2
};

#if 1
class MyTimer {
 public:
  MyTimer()     { dt = dt.zero();                                      }  /**< Clear the duration to 0. */
  void start()  { t0 = std::chrono::high_resolution_clock::now();      }  /**< Start  the timer. */
  void pause()  { dt = std::chrono::high_resolution_clock::now() - t0; }  /**< Pause  the timer. */
  void resume() { t0 = std::chrono::high_resolution_clock::now() - dt; }  /**< Resume the timer. */
  void stop()   { pause();                                             }  /**< Stop   the timer. */
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
  // calendarTime << std::put_time(
  //     &brokenTime,
  //     "%Y-%m-%d-%H-%M-%S");  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
  char time_string[24];
  if (0 < strftime(time_string, sizeof(time_string), "%Y-%m-%d-%H-%M-%S] ", &brokenTime))
    calendarTime << time_string;  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
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
    errNone           = BodyEngine::Err::errNone,
    errGeneral        = BodyEngine::Err::errGeneral,
    errRun            = BodyEngine::Err::errRun,
    errInitialization = BodyEngine::Err::errInitialization,
    errRead           = BodyEngine::Err::errRead,
    errEffect         = BodyEngine::Err::errEffect,
    errParameter      = BodyEngine::Err::errParameter,
    errUnimplemented,
    errMissing,
    errVideo,
    errImageSize,
    errNotFound,
    errBodyModelInit,
    errGLFWInit,
    errGLInit,
    errRendererInit,
    errGLResource,
    errGLGeneric,
    errBodyFit,
    errNoBody,
    errSDK,
    errCuda,
    errCancel,
    errCamera
  };
  Err doAppErr(BodyEngine::Err status) { return (Err)status; }
  BodyEngine body_ar_engine;
  DoApp();
  ~DoApp();

  void stop();
  Err initBodyEngine(const char *modelPath = nullptr);
  Err initCamera(const char *camRes = nullptr);
  Err initOfflineMode(const char *inputFilename = nullptr, const char *outputFilename = nullptr);
  Err acquireFrame();
  Err acquireBodyBox();
  Err acquireBodyBoxAndKeyPoints();
  Err run();
  void drawFPS(cv::Mat &img);
  void DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox);
  //TODO: Look into ways of simplifying the app for these functions.
  void DrawBBoxes(const cv::Mat &src, NvAR_BBoxes *output_bbox);
  void DrawBBoxes(const cv::Mat &src, NvAR_TrackingBBoxes *output_bbox);
  void DrawKeyPointLine(const cv::Mat& src, NvAR_Point2f* keypoints, int point1, int point2, int color);
  void DrawKeyPointsAndEdges(const cv::Mat &src, NvAR_Point2f *keypoints, int numKeyPoints, NvAR_BBoxes* output_bbox);
  void drawKalmanStatus(cv::Mat &img);
  void drawVideoCaptureStatus(cv::Mat &img);
  void processKey(int key);
  void writeVideoAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
  void writeFrameAndEstResults(const cv::Mat &frame, NvAR_BBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
  void writeEstResults(std::ofstream &outputFile, NvAR_BBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
#if NV_MULTI_OBJECT_TRACKER
  void DrawKeyPointsAndEdges(const cv::Mat &src, NvAR_Point2f *keypoints, int numKeyPoints, NvAR_TrackingBBoxes* output_bbox);
  void writeVideoAndEstResults(const cv::Mat &frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
  void writeFrameAndEstResults(const cv::Mat &frame, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
  void writeEstResults(std::ofstream &outputFile, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f *keypoints = NULL);
#endif
  void getFPS();
  static const char *errorStringFromCode(Err code);

  cv::VideoCapture cap{};
  cv::Mat frame;
  int inputWidth, inputHeight;
  cv::VideoWriter bodyDetectOutputVideo{}, keyPointsOutputVideo{};
  int frameIndex;
  static const char windowTitle[];
  double frameTime;
  // std::chrono::high_resolution_clock::time_point frameTimer;
  MyTimer frameTimer;
  cv::VideoWriter capturedVideo;
  std::ofstream bodyEngineVideoOutputFile;

  BodyEngine::Err nvErr;
  float expr[6];
  bool drawVisualization, showFPS, captureVideo, captureFrame;
  float scaleOffsetXY[4];
#if NV_MULTI_OBJECT_TRACKER
  std::vector<cv::Scalar> colorCodes = { cv::Scalar(255,255,255) };
  const unsigned int peopleTrackingBatchSize = 8; // Batch Size has to be 8 when people tracking is enabled
#endif
};

DoApp *gApp = nullptr;
const char DoApp::windowTitle[] = "BodyTrack App";

void DoApp::processKey(int key) {
  switch (key) {
    case '2':
      body_ar_engine.destroyFeatures();
      body_ar_engine.setAppMode(BodyEngine::mode::keyPointDetection);
#if NV_MULTI_OBJECT_TRACKER
      if (FLAG_enablePeopleTracking) body_ar_engine.createFeatures(FLAG_modelPath.c_str());
      else
#endif
         body_ar_engine.createFeatures(FLAG_modelPath.c_str(), 1);
      body_ar_engine.initFeatureIOParams();
      break;
    case '1':
      body_ar_engine.destroyFeatures();
      body_ar_engine.setAppMode(BodyEngine::mode::bodyDetection);
      body_ar_engine.createFeatures(FLAG_modelPath.c_str(), 1);
      body_ar_engine.initFeatureIOParams();
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

DoApp::Err DoApp::initBodyEngine(const char *modelPath) {
  if (!cap.isOpened()) return errVideo;

  int numKeyPoints = body_ar_engine.getNumKeyPoints();
#if NV_MULTI_OBJECT_TRACKER 
  if (FLAG_enablePeopleTracking) nvErr = body_ar_engine.createFeatures(modelPath, peopleTrackingBatchSize);
  else 
#endif
      nvErr = body_ar_engine.createFeatures(modelPath, 1);

#ifdef DEBUG
  detector->setOutputLocation(outputDir);
#endif  // DEBUG

#define VISUALIZE
#ifdef VISUALIZE
  if (!FLAG_offlineMode) cv::namedWindow(windowTitle, 1);
#endif  // VISUALIZE

  frameIndex = 0;

  return doAppErr(nvErr);
}

void DoApp::stop() {
  body_ar_engine.destroyFeatures();

  if (FLAG_offlineMode) {
    bodyDetectOutputVideo.release();
    keyPointsOutputVideo.release();
  }
  cap.release();
#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}

void DoApp::DrawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox) {
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src.clone();
  else
    frm = src;

  if (output_bbox)
    cv::rectangle(frm, cv::Point(lround(output_bbox->x), lround(output_bbox->y)),
                  cv::Point(lround(output_bbox->x + output_bbox->width), lround(output_bbox->y + output_bbox->height)),
                  cv::Scalar(255, 0, 0), 2);
  if (FLAG_offlineMode) bodyDetectOutputVideo.write(frm);
}

void DoApp::DrawBBoxes(const cv::Mat &src, NvAR_BBoxes *output_bbox) {
    cv::Mat frm;
    if (FLAG_offlineMode)
        frm = src.clone();
    else
        frm = src;

    if (output_bbox) {
        for (int i = 0; i < output_bbox->num_boxes; i++) {
            auto color = cv::Scalar(255, 255, 255);
            cv::rectangle(frm, cv::Point(lround(output_bbox->boxes[i].x), lround(output_bbox->boxes[i].y)),
                cv::Point(lround(output_bbox->boxes[i].x + output_bbox->boxes[i].width), lround(output_bbox->boxes[i].y + output_bbox->boxes[i].height)),
                cv::Scalar(255, 0, 0), 2);	
        }
        
    }
        
    if (FLAG_offlineMode) bodyDetectOutputVideo.write(frm);
}
#if NV_MULTI_OBJECT_TRACKER
void DoApp::DrawBBoxes(const cv::Mat &src, NvAR_TrackingBBoxes *output_bbox) {
    cv::Mat frm;
    if (FLAG_offlineMode)
        frm = src.clone();
    else
        frm = src;

    if (output_bbox) {
        for (int i = 0; i < output_bbox->num_boxes; i++) {
            if (colorCodes.size() <= output_bbox->boxes[i].tracking_id)
                colorCodes.push_back(cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF));
            auto color = colorCodes[output_bbox->boxes[i].tracking_id];
            std::string text = "ID: " + std::to_string(output_bbox->boxes[i].tracking_id);
            
            cv::rectangle(frm, cv::Point(lround(output_bbox->boxes[i].bbox.x), lround(output_bbox->boxes[i].bbox.y)),
                cv::Point(lround(output_bbox->boxes[i].bbox.x + output_bbox->boxes[i].bbox.width), lround(output_bbox->boxes[i].bbox.y + output_bbox->boxes[i].bbox.height)),
                color, 2);
            cv::putText(frm, text, cv::Point(lround(output_bbox->boxes[i].bbox.x), lround(output_bbox->boxes[i].bbox.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
            }

    }

    if (FLAG_offlineMode) bodyDetectOutputVideo.write(frm);
}
#endif
void DoApp::writeVideoAndEstResults(const cv::Mat &frm, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints) {
  if (captureVideo) {
    if (!capturedVideo.isOpened()) {
      const std::string currentCalendarTime = getCalendarTime();
      const std::string capturedOutputFileName = currentCalendarTime + ".mp4";
      getFPS();
      if (frameTime) {
        float fps = (float)(1.0 / frameTime);
        capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps,
                           cv::Size(frm.cols, frm.rows));
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
      bodyEngineVideoOutputFile.open(outputsFileName, std::ios_base::out);
      if (!bodyEngineVideoOutputFile.is_open()) {
        std::cout << "Error: Could not open file: \"" << outputsFileName << "\"\n";
        return;
      }
      std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
      bodyEngineVideoOutputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n ";
      bodyEngineVideoOutputFile
          << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
    }
    // Write each frame to the Video
    capturedVideo << frm;
    writeEstResults(bodyEngineVideoOutputFile, output_bboxes, keypoints);
  } else {
    if (capturedVideo.isOpened()) {
      if (FLAG_verbose) {
        std::cout << "Capturing video ended" << std::endl;
      }
      capturedVideo.release();
      if (bodyEngineVideoOutputFile.is_open()) bodyEngineVideoOutputFile.close();
    }
  }
}
#if NV_MULTI_OBJECT_TRACKER
void DoApp::writeVideoAndEstResults(const cv::Mat &frm, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
    if (captureVideo) {
        if (!capturedVideo.isOpened()) {
            const std::string currentCalendarTime = getCalendarTime();
            const std::string capturedOutputFileName = currentCalendarTime + ".mp4";
            getFPS();
            if (frameTime) {
                float fps = (float)(1.0 / frameTime);
                capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps,
                    cv::Size(frm.cols, frm.rows));
                if (!capturedVideo.isOpened()) {
                    std::cout << "Error: Could not open video: \"" << capturedOutputFileName << "\"\n";
                    return;
                }
                if (FLAG_verbose) {
                    std::cout << "Capturing video started" << std::endl;
                }
            }
            else {  // If frameTime is 0.f, returns without writing the frame to the Video
                return;
            }
            const std::string outputsFileName = currentCalendarTime + ".txt";
            bodyEngineVideoOutputFile.open(outputsFileName, std::ios_base::out);
            if (!bodyEngineVideoOutputFile.is_open()) {
                std::cout << "Error: Could not open file: \"" << outputsFileName << "\"\n";
                return;
            }
            std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
            bodyEngineVideoOutputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n ";
            bodyEngineVideoOutputFile
                << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
        }
        // Write each frame to the Video
        capturedVideo << frm;
        writeEstResults(bodyEngineVideoOutputFile, output_bboxes, keypoints);
    }
    else {
        if (capturedVideo.isOpened()) {
            if (FLAG_verbose) {
                std::cout << "Capturing video ended" << std::endl;
            }
            capturedVideo.release();
            if (bodyEngineVideoOutputFile.is_open()) bodyEngineVideoOutputFile.close();
        }
    }
}
#endif
void DoApp::writeEstResults(std::ofstream &outputFile, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints) {
  /**
   * Output File Format :
   * BodyDetectOn, KeyPointDetectOn
   * kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumKPs, [j_x, j_y]{kNumKPs}
   */

  int bodyDetectOn = (body_ar_engine.appMode == BodyEngine::mode::bodyDetection ||
                      body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
                         ? 1
                         : 0;
  int keyPointDetectOn = (body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
                             ? 1
                             : 0;
  outputFile << bodyDetectOn << "," << keyPointDetectOn << "\n";

  if (bodyDetectOn && output_bboxes.num_boxes) {
    // Append number of bodies detected in the current frame
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
  if (keyPointDetectOn && output_bboxes.num_boxes) {
    int numKeyPoints = body_ar_engine.getNumKeyPoints();
    // Append number of keypoints
    outputFile << numKeyPoints << ",";
    // Append 2 * number of keypoint values
    NvAR_Point2f *pt, *endPt;
    for (endPt = (pt = (NvAR_Point2f *)keypoints) + numKeyPoints; pt < endPt; ++pt)
      outputFile << pt->x << "," << pt->y << ",";
  } else {
    outputFile << "0,";
  }

  outputFile << "\n";
}
#if NV_MULTI_OBJECT_TRACKER
void DoApp::writeEstResults(std::ofstream &outputFile, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
    /**
     * Output File Format :
     * BodyDetectOn, KeyPointDetectOn
     * kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumKPs, [j_x, j_y]{kNumKPs}
     */

    int bodyDetectOn = (body_ar_engine.appMode == BodyEngine::mode::bodyDetection ||
        body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
        ? 1
        : 0;
    int keyPointDetectOn = (body_ar_engine.appMode == BodyEngine::mode::keyPointDetection)
        ? 1
        : 0;
    outputFile << bodyDetectOn << "," << keyPointDetectOn << "\n";

    if (bodyDetectOn && output_bboxes.num_boxes) {
        // Append number of bodies detected in the current frame
        outputFile << unsigned(output_bboxes.num_boxes) << ",";
        // write outputbboxes to outputFile
        for (size_t i = 0; i < output_bboxes.num_boxes; i++) {
            int x1 = (int)output_bboxes.boxes[i].bbox.x, y1 = (int)output_bboxes.boxes[i].bbox.y,
                width = (int)output_bboxes.boxes[i].bbox.width, height = (int)output_bboxes.boxes[i].bbox.height;
            unsigned int tracking_id = output_bboxes.boxes[i].tracking_id;
            outputFile << x1 << "," << y1 << "," << width << "," << height << "," << tracking_id << ",";
        }
    }
    else {
        outputFile << "0,";
    }
    if (keyPointDetectOn && output_bboxes.num_boxes) {
        int numKeyPoints = body_ar_engine.getNumKeyPoints();
        // Append number of keypoints
        outputFile << numKeyPoints << ",";
        // Append 2 * number of keypoint values
        NvAR_Point2f *pt, *endPt;
        for (endPt = (pt = (NvAR_Point2f *)keypoints) + numKeyPoints; pt < endPt; ++pt)
            outputFile << pt->x << "," << pt->y << ",";
    }
    else {
        outputFile << "0,";
    }

    outputFile << "\n";
}
#endif
void DoApp::writeFrameAndEstResults(const cv::Mat &frm, NvAR_BBoxes output_bboxes, NvAR_Point2f* keypoints) {
  if (captureFrame) {
    const std::string currentCalendarTime = getCalendarTime();
    const std::string capturedFrame = currentCalendarTime + ".png";
    cv::imwrite(capturedFrame, frm);
    if (FLAG_verbose) {
      std::cout << "Captured the frame" << std::endl;
    }
    // Write Body Engine Outputs
    const std::string outputFilename = currentCalendarTime + ".txt";
    std::ofstream outputFile;
    outputFile.open(outputFilename, std::ios_base::out);
    if (!outputFile.is_open()) {
      std::cout << "Error: Could not open file: \"" << outputFilename << "\"\n";
      return;
    }
    std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
    outputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n";
    outputFile << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
    writeEstResults(outputFile, output_bboxes, keypoints);
    if (outputFile.is_open()) outputFile.close();
    captureFrame = false;
  }
}
#if NV_MULTI_OBJECT_TRACKER
void DoApp::writeFrameAndEstResults(const cv::Mat &frm, NvAR_TrackingBBoxes output_bboxes, NvAR_Point2f* keypoints) {
    if (captureFrame) {
        const std::string currentCalendarTime = getCalendarTime();
        const std::string capturedFrame = currentCalendarTime + ".png";
        cv::imwrite(capturedFrame, frm);
        if (FLAG_verbose) {
            std::cout << "Captured the frame" << std::endl;
        }
        // Write Body Engine Outputs
        const std::string outputFilename = currentCalendarTime + ".txt";
        std::ofstream outputFile;
        outputFile.open(outputFilename, std::ios_base::out);
        if (!outputFile.is_open()) {
            std::cout << "Error: Could not open file: \"" << outputFilename << "\"\n";
            return;
        }
        std::string keyPointDetectionMode = (keypoints == NULL) ? "Off" : "On";
        outputFile << "// BodyDetectOn, KeyPointDetect" << keyPointDetectionMode << "\n";
        outputFile << "// kNumPeople, (bbox_x, bbox_y, bbox_w, bbox_h){ kNumPeople}, kNumLMs, [lm_x, lm_y]{kNumLMs}\n";
        writeEstResults(outputFile, output_bboxes, keypoints);
        if (outputFile.is_open()) outputFile.close();
        captureFrame = false;
    }
}
#endif
void DoApp::DrawKeyPointLine(const cv::Mat& src, NvAR_Point2f* keypoints, int point1, int point2, int color) {
  NvAR_Point2f point1_pos = *(keypoints + point1);
  NvAR_Point2f point2_pos = *(keypoints + point2);
  cv::line(src, cv::Point((int)point1_pos.x, (int)point1_pos.y), cv::Point((int)point2_pos.x, (int)point2_pos.y), cv_colors[color], 2);

}
#if NV_MULTI_OBJECT_TRACKER
void DoApp::DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_TrackingBBoxes* output_bbox) {
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src.clone();
  else
    frm = src;
  NvAR_Point2f *pt, *endPt;
  NvAR_Point2f* keypointsBatch8 = keypoints;

  int pelvis = 0;
  int left_hip = 1;
  int right_hip = 2;
  int torso = 3;
  int left_knee = 4;
  int right_knee = 5;
  int neck = 6;
  int left_ankle = 7;
  int right_ankle = 8;
  int left_big_toe = 9;
  int right_big_toe = 10;
  int left_small_toe = 11;
  int right_small_toe = 12;
  int left_heel = 13;
  int right_heel = 14;
  int nose = 15;
  int left_eye = 16;
  int right_eye = 17;
  int left_ear = 18;
  int right_ear = 19;
  int left_shoulder = 20;
  int right_shoulder = 21;
  int left_elbow = 22;
  int right_elbow = 23;
  int left_wrist = 24;
  int right_wrist = 25;
  int left_pinky_knuckle = 26;
  int right_pinky_knuckle = 27;
  int left_middle_tip = 28;
  int right_middle_tip = 29;
  int left_index_knuckle = 30;
  int right_index_knuckle = 31;
  int left_thumb_tip = 32;
  int right_thumb_tip = 33;

  for (int i = 0; i < body_ar_engine.output_tracking_bboxes.num_boxes; i++) {

      keypoints = keypointsBatch8 + (i * 34);

      for (endPt = (pt = (NvAR_Point2f*)keypoints) + numKeyPoints; pt < endPt; ++pt)
          cv::circle(frm, cv::Point(lround(pt->x), lround(pt->y)), 4, cv::Scalar(180, 180, 180), -1);

      if (output_bbox) {
          while (colorCodes.size()<= output_bbox->boxes[i].tracking_id)
                  colorCodes.push_back(cv::Scalar(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF));
          auto color = colorCodes[output_bbox->boxes[i].tracking_id];
          std::string text = "ID: "+std::to_string(output_bbox->boxes[i].tracking_id);
          cv::rectangle(frm, cv::Point(lround(output_bbox->boxes[i].bbox.x), lround(output_bbox->boxes[i].bbox.y)),
              cv::Point(lround(output_bbox->boxes[i].bbox.x + output_bbox->boxes[i].bbox.width), lround(output_bbox->boxes[i].bbox.y + output_bbox->boxes[i].bbox.height)),
              color, 2);
          cv::putText(frm, text, cv::Point(lround(output_bbox->boxes[i].bbox.x), lround(output_bbox->boxes[i].bbox.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
      }
          

      // center body
      DrawKeyPointLine(frm, keypoints, pelvis, torso, kColorGreen);
      DrawKeyPointLine(frm, keypoints, torso, neck, kColorGreen);
      DrawKeyPointLine(frm, keypoints, neck, pelvis, kColorGreen);

      // right side
      DrawKeyPointLine(frm, keypoints, right_ankle, right_knee, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_knee, right_hip, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_hip, pelvis, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_hip, right_shoulder, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_shoulder, right_elbow, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_elbow, right_wrist, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_shoulder, neck, kColorRed);

      // right side hand and feet
      DrawKeyPointLine(frm, keypoints, right_wrist, right_pinky_knuckle, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_wrist, right_middle_tip, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_wrist, right_index_knuckle, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_wrist, right_thumb_tip, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_ankle, right_heel, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_ankle, right_big_toe, kColorRed);
      DrawKeyPointLine(frm, keypoints, right_big_toe, right_small_toe, kColorRed);

      //left side
      DrawKeyPointLine(frm, keypoints, left_ankle, left_knee, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_knee, left_hip, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_hip, pelvis, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_hip, left_shoulder, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_shoulder, left_elbow, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_elbow, left_wrist, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_shoulder, neck, kColorBlue);

      // left side hand and feet
      DrawKeyPointLine(frm, keypoints, left_wrist, left_pinky_knuckle, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_wrist, left_middle_tip, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_wrist, left_index_knuckle, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_wrist, left_thumb_tip, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_ankle, left_heel, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_ankle, left_big_toe, kColorBlue);
      DrawKeyPointLine(frm, keypoints, left_big_toe, left_small_toe, kColorBlue);

      // head
      DrawKeyPointLine(frm, keypoints, neck, nose, kColorGreen);
      DrawKeyPointLine(frm, keypoints, nose, right_eye, kColorGreen);
      DrawKeyPointLine(frm, keypoints, right_eye, right_ear, kColorGreen);
      DrawKeyPointLine(frm, keypoints, nose, left_eye, kColorGreen);
      DrawKeyPointLine(frm, keypoints, left_eye, left_ear, kColorGreen);
  }

  if (FLAG_offlineMode) keyPointsOutputVideo.write(frm);
}
#endif
void DoApp::DrawKeyPointsAndEdges(const cv::Mat& src, NvAR_Point2f* keypoints, int numKeyPoints, NvAR_BBoxes* output_bbox) {
    cv::Mat frm;
    if (FLAG_offlineMode)
        frm = src.clone();
    else
        frm = src;
    NvAR_Point2f *pt, *endPt;
    NvAR_Point2f* keypointsBatch8 = keypoints;

    int pelvis = 0;
    int left_hip = 1;
    int right_hip = 2;
    int torso = 3;
    int left_knee = 4;
    int right_knee = 5;
    int neck = 6;
    int left_ankle = 7;
    int right_ankle = 8;
    int left_big_toe = 9;
    int right_big_toe = 10;
    int left_small_toe = 11;
    int right_small_toe = 12;
    int left_heel = 13;
    int right_heel = 14;
    int nose = 15;
    int left_eye = 16;
    int right_eye = 17;
    int left_ear = 18;
    int right_ear = 19;
    int left_shoulder = 20;
    int right_shoulder = 21;
    int left_elbow = 22;
    int right_elbow = 23;
    int left_wrist = 24;
    int right_wrist = 25;
    int left_pinky_knuckle = 26;
    int right_pinky_knuckle = 27;
    int left_middle_tip = 28;
    int right_middle_tip = 29;
    int left_index_knuckle = 30;
    int right_index_knuckle = 31;
    int left_thumb_tip = 32;
    int right_thumb_tip = 33;

    for (int i = 0; i < body_ar_engine.output_bboxes.num_boxes; i++) {

        keypoints = keypointsBatch8 + (i * 34);

        for (endPt = (pt = (NvAR_Point2f*)keypoints) + numKeyPoints; pt < endPt; ++pt)
            cv::circle(frm, cv::Point(lround(pt->x), lround(pt->y)), 4, cv::Scalar(180, 180, 180), -1);

        if (output_bbox) {
            cv::rectangle(frm, cv::Point(lround(output_bbox->boxes[i].x), lround(output_bbox->boxes[i].y)),
                cv::Point(lround(output_bbox->boxes[i].x + output_bbox->boxes[i].width), lround(output_bbox->boxes[i].y + output_bbox->boxes[i].height)),
                cv::Scalar(255, 0, 0), 2);
        }


        // center body
        DrawKeyPointLine(frm, keypoints, pelvis, torso, kColorGreen);
        DrawKeyPointLine(frm, keypoints, torso, neck, kColorGreen);
        DrawKeyPointLine(frm, keypoints, neck, pelvis, kColorGreen);

        // right side
        DrawKeyPointLine(frm, keypoints, right_ankle, right_knee, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_knee, right_hip, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_hip, pelvis, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_hip, right_shoulder, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_shoulder, right_elbow, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_elbow, right_wrist, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_shoulder, neck, kColorRed);

        // right side hand and feet
        DrawKeyPointLine(frm, keypoints, right_wrist, right_pinky_knuckle, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_wrist, right_middle_tip, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_wrist, right_index_knuckle, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_wrist, right_thumb_tip, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_ankle, right_heel, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_ankle, right_big_toe, kColorRed);
        DrawKeyPointLine(frm, keypoints, right_big_toe, right_small_toe, kColorRed);

        //left side
        DrawKeyPointLine(frm, keypoints, left_ankle, left_knee, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_knee, left_hip, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_hip, pelvis, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_hip, left_shoulder, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_shoulder, left_elbow, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_elbow, left_wrist, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_shoulder, neck, kColorBlue);

        // left side hand and feet
        DrawKeyPointLine(frm, keypoints, left_wrist, left_pinky_knuckle, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_wrist, left_middle_tip, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_wrist, left_index_knuckle, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_wrist, left_thumb_tip, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_ankle, left_heel, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_ankle, left_big_toe, kColorBlue);
        DrawKeyPointLine(frm, keypoints, left_big_toe, left_small_toe, kColorBlue);

        // head
        DrawKeyPointLine(frm, keypoints, neck, nose, kColorGreen);
        DrawKeyPointLine(frm, keypoints, nose, right_eye, kColorGreen);
        DrawKeyPointLine(frm, keypoints, right_eye, right_ear, kColorGreen);
        DrawKeyPointLine(frm, keypoints, nose, left_eye, kColorGreen);
        DrawKeyPointLine(frm, keypoints, left_eye, left_ear, kColorGreen);

    }

    if (FLAG_offlineMode) keyPointsOutputVideo.write(frm);
}

DoApp::Err DoApp::acquireFrame() {
  Err err = errNone;

  // If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
  // frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
  // resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
  // done here) as well as reallocate memory accordingly with BodyEngine::initFeatureIOParams()
  cap >> frame;  // get a new frame from camera into the class variable frame.
  if (frame.empty()) {
    // if in Offline mode, this means end of video,so we return
    if (FLAG_offlineMode) return errVideo;
    // try Init one more time if reading frames from camera
    err = initCamera(FLAG_camRes.c_str());
    if (err != errNone)
      return err;
    cap >> frame;
    if (frame.empty()) return errVideo;
  }

  return err;
}

DoApp::Err DoApp::acquireBodyBox() {
  Err err = errNone;
  NvAR_Rect output_bbox;

  // get keypoints in  original image resolution coordinate space
  unsigned n = body_ar_engine.acquireBodyBox(frame, output_bbox, 0);

  if (n && FLAG_verbose) {
    printf("BodyBox: [\n");
    printf("%7.1f%7.1f%7.1f%7.1f\n", output_bbox.x, output_bbox.y, output_bbox.x + output_bbox.width,
           output_bbox.y + output_bbox.height);
    printf("]\n");
  }
  if (FLAG_captureOutputs) {
#if NV_MULTI_OBJECT_TRACKER
      if (FLAG_enablePeopleTracking) {
          writeFrameAndEstResults(frame, body_ar_engine.output_tracking_bboxes);
          writeVideoAndEstResults(frame, body_ar_engine.output_tracking_bboxes);
      }
      else {
          writeFrameAndEstResults(frame, body_ar_engine.output_bboxes);
          writeVideoAndEstResults(frame, body_ar_engine.output_bboxes);
      }
#else
      writeFrameAndEstResults(frame, body_ar_engine.output_bboxes);
      writeVideoAndEstResults(frame, body_ar_engine.output_bboxes);
#endif
    
  }
  if (0 == n) return errNoBody;

#ifdef VISUALIZE

  if (drawVisualization) {
    DrawBBoxes(frame, &output_bbox);
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::acquireBodyBoxAndKeyPoints() {
  Err err = errNone;
  int numKeyPoints = body_ar_engine.getNumKeyPoints();
  NvAR_BBoxes output_bbox;
  NvAR_TrackingBBoxes output_tracking_bbox;
  std::vector<NvAR_Point2f> keypoints2D(numKeyPoints * 8);
  std::vector<NvAR_Point3f> keypoints3D(numKeyPoints * 8);
  std::vector<NvAR_Quaternion> jointAngles(numKeyPoints * 8);


#ifdef DEBUG_PERF_RUNTIME
  auto start = std::chrono::high_resolution_clock::now();
#endif

  unsigned n;
  // get keypoints in original image resolution coordinate space
#if NV_MULTI_OBJECT_TRACKER
  if (FLAG_enablePeopleTracking) 
      n = body_ar_engine.acquireBodyBoxAndKeyPoints(frame, keypoints2D.data(), keypoints3D.data(),
          jointAngles.data(), &output_tracking_bbox, 0);
  
  else 
#endif
      n = body_ar_engine.acquireBodyBoxAndKeyPoints(frame, keypoints2D.data(), keypoints3D.data(),
          jointAngles.data(), &output_bbox, 0);
  

#ifdef DEBUG_PERF_RUNTIME
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "box+keypoints time: " << duration.count() << " microseconds" << std::endl;
#endif

  if (n && FLAG_verbose && body_ar_engine.appMode != BodyEngine::mode::bodyDetection) {
    printf("KeyPoints: [\n");
    for (const auto &pt : keypoints2D) {
      printf("%7.1f%7.1f\n", pt.x, pt.y);
    }
    printf("]\n");

    printf("3d KeyPoints: [\n");
    for (const auto& pt : keypoints3D) {
        printf("%7.1f%7.1f%7.1f\n", pt.x, pt.y, pt.z);
    }
    printf("]\n");
  }
  if (FLAG_captureOutputs) {
#if NV_MULTI_OBJECT_TRACKER
      if (FLAG_enablePeopleTracking) {
          writeFrameAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
          writeVideoAndEstResults(frame, body_ar_engine.output_tracking_bboxes, keypoints2D.data());
      }
      else {
          writeFrameAndEstResults(frame, body_ar_engine.output_bboxes, keypoints2D.data());
          writeVideoAndEstResults(frame, body_ar_engine.output_bboxes, keypoints2D.data());
      }
#else
      writeFrameAndEstResults(frame, body_ar_engine.output_bboxes, keypoints2D.data());
      writeVideoAndEstResults(frame, body_ar_engine.output_bboxes, keypoints2D.data());
#endif
    
  }
  if (0 == n) return errNoBody;

#ifdef VISUALIZE

  if (drawVisualization) {
#if NV_MULTI_OBJECT_TRACKER
      if (FLAG_enablePeopleTracking) DrawKeyPointsAndEdges(frame, keypoints2D.data(), numKeyPoints, &output_tracking_bbox);
      else 
#endif
          DrawKeyPointsAndEdges(frame, keypoints2D.data(), numKeyPoints, &output_bbox);
    if (FLAG_offlineMode) {
#if NV_MULTI_OBJECT_TRACKER
        if (FLAG_enablePeopleTracking) DrawBBoxes(frame, &output_tracking_bbox);
        else 
#endif
            DrawBBoxes(frame, &output_bbox);
    }
  }
#endif  // VISUALIZE
  frameIndex++;

  return err;
}

DoApp::Err DoApp::initCamera(const char *camRes) {
  if (cap.open(FLAG_camindex)) {
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

      inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
      inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
      body_ar_engine.setInputImageWidth(inputWidth);
      body_ar_engine.setInputImageHeight(inputHeight);
    }
  } else
    return errCamera;
  return errNone;
}

DoApp::Err DoApp::initOfflineMode(const char *inputFilename, const char *outputFilename) {
  if (cap.open(inputFilename)) {
    inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    body_ar_engine.setInputImageWidth(inputWidth);
    body_ar_engine.setInputImageHeight(inputHeight);
  } else {
    printf("ERROR: Unable to open the input video file \"%s\" \n", inputFilename);
    return Err::errVideo;
  }

  std::string bdOutputVideoName, jdOutputVideoName;
  std::string outputFilePrefix;
  if (outputFilename && strlen(outputFilename) != 0) {
    outputFilePrefix = outputFilename;
  } else {
    size_t lastindex = std::string(inputFilename).find_last_of(".");
    outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
  }
  bdOutputVideoName = outputFilePrefix + "_bbox.mp4";
  jdOutputVideoName = outputFilePrefix + "_pose.mp4";

  if (!bodyDetectOutputVideo.open(bdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                  cv::Size(inputWidth, inputHeight))) {
    printf("ERROR: Unable to open the output video file \"%s\" \n", bdOutputVideoName.c_str());
    return Err::errGeneral;
  }
  if (!keyPointsOutputVideo.open(jdOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
      cv::Size(inputWidth, inputHeight))) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", bdOutputVideoName.c_str());
      return Err::errGeneral;
  }

  return Err::errNone;
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
  nvErr = BodyEngine::errNone;
  scaleOffsetXY[0] = scaleOffsetXY[2] = 1.f;
  scaleOffsetXY[1] = scaleOffsetXY[3] = 0.f;
}

DoApp::~DoApp() {}

char *g_nvARSDKPath = NULL;

int chooseGPU() {
  // If the system has multiple supported GPUs then the application
  // should use CUDA driver APIs or CUDA runtime APIs to enumerate
  // the GPUs and select one based on the application's requirements

  //Cuda device 0
  return 0;

}

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
  snprintf(buf, sizeof(buf), "Kalman %s", (body_ar_engine.bStabilizeBody ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

void DoApp::drawVideoCaptureStatus(cv::Mat &img) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Video Capturing %s", (captureVideo ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

DoApp::Err DoApp::run() {
  DoApp::Err doErr = errNone;

  BodyEngine::Err err = body_ar_engine.initFeatureIOParams();
  if (err != BodyEngine::Err::errNone ) {
    return doAppErr(err);
  }
  while (1) {
    //printf(">> frame %d \n", framenum++);
    doErr = acquireFrame();
    if (frame.empty() && FLAG_offlineMode) {
      // We have reached the end of the video
      // so return without any error.
      return DoApp::errNone;
    }
    else if (doErr != DoApp::errNone) {
      return doErr;
    }
    if (body_ar_engine.appMode == BodyEngine::mode::bodyDetection) {
      doErr = acquireBodyBox();
    } else if (body_ar_engine.appMode == BodyEngine::mode::keyPointDetection) {
      doErr = acquireBodyBoxAndKeyPoints();
    }
    if ((DoApp::errNoBody == doErr || DoApp::errBodyFit == doErr) && FLAG_offlineMode) {
      bodyDetectOutputVideo.write(frame);
      keyPointsOutputVideo.write(frame);
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

const char *DoApp::errorStringFromCode(DoApp::Err code) {
  struct LUTEntry {
    Err code;
    const char *str;
  };
  static const LUTEntry lut[] = {
      {errNone, "no error"},
      {errGeneral, "an error has occured"},
      {errRun, "an error has occured while the feature is running"},
      {errInitialization, "Initializing Body Engine failed"},
      {errRead, "an error has occured while reading a file"},
      {errEffect, "an error has occured while creating a feature"},
      {errParameter, "an error has occured while setting a parameter for a feature"},
      {errUnimplemented, "the feature is unimplemented"},
      {errMissing, "missing input parameter"},
      {errVideo, "no video source has been found"},
      {errImageSize, "the image size cannot be accommodated"},
      {errNotFound, "the item cannot be found"},
      {errBodyModelInit, "body model initialization failed"},
      {errGLFWInit, "GLFW initialization failed"},
      {errGLInit, "OpenGL initialization failed"},
      {errRendererInit, "renderer initialization failed"},
      {errGLResource, "an OpenGL resource could not be found"},
      {errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
      {errBodyFit, "an error has occurred while body fitting"},
      {errNoBody, "no body has been found"},
      {errSDK, "an SDK error has occurred"},
      {errCuda, "a CUDA error has occurred"},
      {errCancel, "the user cancelled"},
      {errCamera, "unable to connect to the camera"},
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
  // Parse the arguments
  if (0 != ParseMyArgs(argc, argv)) return -100;

  DoApp app;
  DoApp::Err doErr = DoApp::Err::errNone;

  app.body_ar_engine.setAppMode(BodyEngine::mode(FLAG_appMode));

  app.body_ar_engine.setMode(FLAG_mode);

  if (FLAG_verbose) printf("Enable temporal optimizations in detecting body and keypoints = %d\n", FLAG_temporal);
  app.body_ar_engine.setBodyStabilization(FLAG_temporal);

  if (FLAG_useCudaGraph) printf("Enable capturing cuda graph = %d\n", FLAG_useCudaGraph);
  app.body_ar_engine.useCudaGraph(FLAG_useCudaGraph);
#if NV_MULTI_OBJECT_TRACKER
  app.body_ar_engine.enablePeopleTracking(FLAG_enablePeopleTracking, FLAG_shadowTrackingAge, FLAG_probationAge, FLAG_maxTargetsTracked);
#endif
  doErr = DoApp::errBodyModelInit;
  if (FLAG_modelPath.empty()) {
    printf("WARNING: Model path not specified. Please set --model_path=/path/to/trt/and/body/models, "
      "SDK will attempt to load the models from NVAR_MODEL_DIR environment variable, "
      "please restart your application after the SDK Installation. \n");
  }
  if (!FLAG_bodyModel.empty())
    app.body_ar_engine.setBodyModel(FLAG_bodyModel.c_str());

  if (FLAG_offlineMode) {
    if (FLAG_inFile.empty()) {
      doErr = DoApp::errMissing;
      printf("ERROR: %s, please specify input file using --in_file or --in \n", app.errorStringFromCode(doErr));
      goto bail;
    }
    doErr = app.initOfflineMode(FLAG_inFile.c_str(), FLAG_outFile.c_str());
  } else {
    doErr = app.initCamera(FLAG_camRes.c_str());
  }
  BAIL_IF_ERR(doErr);

  doErr = app.initBodyEngine(FLAG_modelPath.c_str());
  BAIL_IF_ERR(doErr);

  doErr = app.run();
  BAIL_IF_ERR(doErr);

bail:
  if(doErr)
    printf("ERROR: %s\n", app.errorStringFromCode(doErr));
  app.stop();
  return (int)doErr;
}
