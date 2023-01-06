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
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "GazeEngine.h"
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
#define DEGREES_PER_RADIAN (180.0 / M_PI)

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
     FLAG_drawVisualization = true, FLAG_offlineMode = false, FLAG_isNumLandmarks126 = false,
     FLAG_splitScreenView = true, FLAG_displayLandmarks = false, FLAG_gazeRedirect = true, FLAG_useCudaGraph = false;
;
std::string FLAG_outDir, FLAG_inFile, FLAG_outFile, FLAG_modelPath, FLAG_landmarks, FLAG_captureCodec = "avc1",
                                                                                    FLAG_camRes = "480";
unsigned FLAG_camID = 0;
unsigned FLAG_eyeSizeSensitivity = 3;

/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  printf(
    "GazeRedirect [<args> ...]\n"
    "where <args> is\n"
    " --verbose[=(true|false)]          report interesting info\n"
    " --debug[=(true|false)]            report debugging info\n"
    " --temporal[=(true|false)]         temporally optimize face rect and landmarks\n"
    " --capture_outputs[=(true|false)]  enables video/image capture and writing face detection/landmark outputs\n"
    " --offline_mode[=(true|false)]     disables webcam, reads video from file and writes output video results\n"
    " --cam_res=[WWWx]HHH               specify resolution as height or width x height\n"
    " --cam_id=<id>                     by default 0, specify int ID of camera in case of multiple cameras \n"
    " --codec=<fourcc>                  FOURCC code for the desired codec (default H264)\n"
    " --in=<file>                       specify the  input file\n"
    " --out=<file>                      specify the output file\n"
    " --model_path=<path>               specify the directory containing the TRT models\n"
    " --landmarks_126[=(true|false)]    set the number of facial landmark points to 126, otherwise default to 68\n"
    " --eyesize_sensitivity             set the eye size sensitivity parameter, an integer value between 2 and 6 (default 3)\n"
    " --split_screen_view               split the screen to view original image side-by-side with the gaze redirected image, default to split. \n"
    " --draw_visualization              draw the landmarks, display gaze estimation and head rotation, default to true\n"
    " --redirect_gaze                   redirection of the eyes in addition to estimating gaze, default to true\n"
    "(Default)."
    " --use_cuda_graph                 use cuda graph to optimize computations "
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
  // query NVAR_MODEL_DIR environment variable first before checking the command line arguments
  const char *modelPath = getenv("NVAR_MODEL_DIR");
  if (modelPath) {
    FLAG_modelPath = modelPath;
  }

  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char *arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') &&
        (GetFlagArgVal("verbose", arg, &FLAG_verbose) || GetFlagArgVal("debug", arg, &FLAG_debug) ||
        GetFlagArgVal("in", arg, &FLAG_inFile) || GetFlagArgVal("out", arg, &FLAG_outFile) ||
        GetFlagArgVal("offline_mode", arg, &FLAG_offlineMode) ||
        GetFlagArgVal("landmarks_126", arg, &FLAG_isNumLandmarks126) ||
        GetFlagArgVal("capture_outputs", arg, &FLAG_captureOutputs) ||
        GetFlagArgVal("cam_res", arg, &FLAG_camRes) || GetFlagArgVal("codec", arg, &FLAG_captureCodec) ||
        GetFlagArgVal("cam_id", arg, &FLAG_camID) ||
        GetFlagArgVal("landmarks", arg, &FLAG_landmarks) || GetFlagArgVal("model_path", arg, &FLAG_modelPath) ||
        GetFlagArgVal("eyesize_sensitivity", arg, &FLAG_eyeSizeSensitivity) ||
        GetFlagArgVal("split_screen_view", arg, &FLAG_splitScreenView) ||
        GetFlagArgVal("temporal", arg, &FLAG_temporal) ||
        GetFlagArgVal("draw_visualization", arg, &FLAG_drawVisualization) ||
        GetFlagArgVal("redirect_gaze", arg, &FLAG_gazeRedirect) ||
        GetFlagArgVal("use_cuda_graph", arg, &FLAG_useCudaGraph))) {
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

  // std::put_time has not been implemented under GCC5. In order to support centOS7(GCC4.8), we use strftime.
  // (YYYY-MM-DD-HH-mm-ss)<Year>-<Month>-<Date>-<Hour>-<Mins>-<Seconds>
  // Get the time since epoch 0(Thu Jan  1 00:00:00 1970) and the remainder after division is our milliseconds
  char foo[24];
  if(0 < strftime(foo, sizeof(foo), "%Y-%m-%d-%H-%M-%S", &brokenTime)) {
    //cout << foo << endl;
    std::string fooStr(foo);
    calendarTime << fooStr;
  }

  std::chrono::milliseconds currentMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(currentTimePoint.time_since_epoch()) % 1000;
  // Append the milliseconds to the stream
  calendarTime << "-" << std::setfill('0') << std::setw(3) << currentMilliseconds.count();  // milliseconds
  return calendarTime.str();
}

class DoApp {
 public:
  enum Err {
    errNone = GazeEngine::Err::errNone,
    errGeneral = GazeEngine::Err::errGeneral,
    errRun = GazeEngine::Err::errRun,
    errInitialization = GazeEngine::Err::errInitialization,
    errRead = GazeEngine::Err::errRead,
    errEffect = GazeEngine::Err::errEffect,
    errParameter = GazeEngine::Err::errParameter,
    errUnimplemented,
    errMissing,
    errVideo,
    errImageSize,
    errNotFound,
    errGLFWInit,
    errGLInit,
    errRendererInit,
    errGLResource,
    errGLGeneric,
    errSDK,
    errCuda,
    errCancel,
    errCamera
  };
  Err doAppErr(GazeEngine::Err status) { return (Err)status; }
  GazeEngine gaze_ar_engine;
  DoApp();
  ~DoApp();

  void stop();
  Err initGazeEngine(const char *modelPath = nullptr, bool isLandmarks126 = false, bool gazeRedirect = true,
                     unsigned eyeSizeSensitivity = 3, bool useCudaGraph = false);
  Err initCamera(const char *camRes = nullptr, unsigned int camID = 0);
  Err initOfflineMode(const char *inputFilename = nullptr, const char *outputFilename = nullptr);
  Err acquireFrame();
  Err acquireFaceBox();
  Err acquireFaceBoxAndLandmarks();
  Err acquireGazeRedirection();
  Err run();

  void drawFPS(cv::Mat &img);
  void drawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox);
  void DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks, int numLandmarks, cv::Scalar *color);
  void drawKalmanStatus(cv::Mat &img);
  void drawVideoCaptureStatus(cv::Mat &img);
  void processKey(int key);
  Err writeVideo(const cv::Mat &frm);
  void getFPS();
  static const char *errorStringFromCode(Err code);

  cv::VideoCapture cap{};
  cv::Mat frame, outputFrame;

  int inputWidth, inputHeight;
  cv::VideoWriter gazeRedirectOutputVideo{};
  int frameIndex;
  static const char windowTitle[];
  double frameTime;
  // std::chrono::high_resolution_clock::time_point frameTimer;
  MyTimer frameTimer;
  cv::VideoWriter capturedVideo;
  std::ofstream gazeEngineVideoOutputFile;

  GazeEngine::Err nvErr;
  bool drawVisualization, showFPS, captureVideo, splitScreenView, displayLandmarks;
};

DoApp *gApp = nullptr;
const char DoApp::windowTitle[] = "GazeRedirect App";

void DoApp::processKey(int key) {
  switch (key) {
    case '3':
      gaze_ar_engine.destroyGazeRedirectionFeature();
      gaze_ar_engine.createGazeRedirectionFeature(FLAG_modelPath.c_str());
      gaze_ar_engine.initGazeRedirectionIOParams();
      break;
    case 'C':
    case 'c':
      captureVideo = !captureVideo;
      break;
    case 'W':
    case 'w':
      drawVisualization = !drawVisualization;
      break;
    case 'F':
    case 'f':
      showFPS = !showFPS;
      break;
    case 'O':
    case 'o':
      splitScreenView = !splitScreenView;
      break;
    case 'L':
    case 'l':
      displayLandmarks = !displayLandmarks;
      break;
    default:
      break;
  }
}

DoApp::Err DoApp::initGazeEngine(const char *modelPath, bool isNumLandmarks126, bool gazeRedirect,
                                 unsigned eyeSizeSensitivity, bool useCudaGraph) {
  if (!cap.isOpened()) return errVideo;

  int numLandmarkPoints = isNumLandmarks126 ? 126 : 68;
  gaze_ar_engine.setNumLandmarks(numLandmarkPoints);
  gaze_ar_engine.setGazeRedirect(gazeRedirect);
  gaze_ar_engine.setUseCudaGraph(useCudaGraph);
  gaze_ar_engine.setEyeSizeSensitivity(eyeSizeSensitivity);
  nvErr = gaze_ar_engine.createGazeRedirectionFeature(modelPath);

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
  gaze_ar_engine.destroyGazeRedirectionFeature();

  if (FLAG_offlineMode) {
    gazeRedirectOutputVideo.release();
  }
  cap.release();
#ifdef VISUALIZE
  cv::destroyAllWindows();
#endif  // VISUALIZE
}

void DoApp::drawBBoxes(const cv::Mat &src, NvAR_Rect *output_bbox) {
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src.clone();
  else
    frm = src;

  if (output_bbox)
    cv::rectangle(frm, cv::Point(lround(output_bbox->x), lround(output_bbox->y)),
                  cv::Point(lround(output_bbox->x + output_bbox->width), lround(output_bbox->y + output_bbox->height)),
                  cv::Scalar(255, 0, 0), 2);
}

DoApp::Err DoApp::writeVideo(const cv::Mat &frm) {
  if (captureVideo) {
    if (!capturedVideo.isOpened()) {
      //Assign the filename for capturing video
      const std::string currentCalendarTime = getCalendarTime();
      const std::string capturedOutputFileName =  currentCalendarTime + ".mp4";
      getFPS();
      if (frameTime) {
        float fps = (float)(1.0 / frameTime);
        // Open the video for writing.
        capturedVideo.open(capturedOutputFileName, StringToFourcc(FLAG_captureCodec), fps,
                           cv::Size(frm.cols, frm.rows));
        if (!capturedVideo.isOpened()) {
          
          std::cout << "Error: Could not open video for writing: \"" << capturedOutputFileName << "\"\n";
          return errVideo;
        }
        if (FLAG_verbose) {
          std::cout << "Capturing video started" << std::endl;
        }
        capturedVideo << frm;
      } else {  // If frameTime is 0.f, returns without writing the frame to the Video
        return errNone;
      }
    } else {
      // Write each frame to the Video
      capturedVideo << frm;
    }
  } else {
    if (capturedVideo.isOpened()) {
      if (FLAG_verbose) {
        std::cout << "Capturing video ended" << std::endl;
      }
      capturedVideo.release();
    }
  }
  return errNone;
}

void DoApp::DrawLandmarkPoints(const cv::Mat &src, NvAR_Point2f *facial_landmarks, int numLandmarks , cv::Scalar* color) {
  if (!facial_landmarks)
    return; 
  cv::Mat frm;
  if (FLAG_offlineMode)
    frm = src;
  else
    frm = src;
  NvAR_Point2f *pt, *endPt;
  for (endPt = (pt = (NvAR_Point2f *)facial_landmarks) + numLandmarks; pt < endPt; ++pt)
    cv::circle(frm, cv::Point(lround(pt->x), lround(pt->y)), 1.5, *color, -1);
}

DoApp::Err DoApp::acquireFrame() {
  Err err = errNone;

  // If the machine goes to sleep with the app running and then wakes up, the camera object is not destroyed but the
  // frames we try to read are empty. So we try to re-initialize the camera with the same resolution settings. If the
  // resolution has changed, you will need to destroy and create the features again with the new camera resolution (not
  // done here) as well as reallocate memory accordingly with GazeEngine::initGazeRedirectionIOParams()
  cap >> frame;  // get a new frame from camera into the class variable frame.
  if (frame.empty()) {
    // if in Offline mode, this means end of video,so we return
    if (FLAG_offlineMode) return errVideo;
    // try Init one more time if reading frames from camera
    err = initCamera(FLAG_camRes.c_str(), FLAG_camID);
    if (err != errNone) return err;
    cap >> frame;
    if (frame.empty()) return errVideo;
  }

  return err;
}

DoApp::Err DoApp::acquireGazeRedirection() {
  DoApp::Err doErr = errNone;

  nvErr = gaze_ar_engine.acquireGazeRedirection(frame, outputFrame);

  if (GazeEngine::Err::errNone == nvErr) {
#ifdef VISUALIZE
    frameTimer.pause();
    NvAR_Rect *bbox = gaze_ar_engine.getLargestBox();
    // Check for valid bounding box in case confidence check fails
    if (drawVisualization && bbox) {
      // Display gaze direction and head rotation
      NvAR_Quaternion *pose = gaze_ar_engine.getPose();
      NvAR_Point3f *gaze_direction = gaze_ar_engine.getGazeDirectionPoints();
      if (pose) {
        gaze_ar_engine.DrawPose(frame, pose);
        gaze_ar_engine.DrawEstimatedGaze(frame);
      }
      // Print head pose
      char buf[64];
      float fontsize;
      if (inputHeight <= 720) {
        fontsize = 0.5;
      } else {
        fontsize = .5;
      }
      snprintf(buf, sizeof(buf), " Original image");
      cv::putText(frame, buf, cv::Point(120, 40), cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 255, 0), 1);
      snprintf(buf, sizeof(buf), "gaze angles: %.1f %.1f", gaze_ar_engine.gaze_angles_vector[0] * DEGREES_PER_RADIAN,
               gaze_ar_engine.gaze_angles_vector[1] * DEGREES_PER_RADIAN);
      cv::putText(frame, buf, cv::Point(120, 110), cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 255, 0), 1);
      // display head translationss
      snprintf(buf, sizeof(buf), "head translation: %.1f %.1f %.1f", gaze_ar_engine.head_translation[0],
               gaze_ar_engine.head_translation[1], gaze_ar_engine.head_translation[2]);
      cv::putText(frame, buf, cv::Point(80, 80), cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 255, 0), 1);
      // Display landmarks if flag is set.
      if (displayLandmarks) {
        cv::Scalar landmarks_color(0, 0, 255);
        DrawLandmarkPoints(frame, gaze_ar_engine.getLandmarks(), gaze_ar_engine.getNumLandmarks(), &landmarks_color);
        drawBBoxes(frame, gaze_ar_engine.getLargestBox());
      }
      snprintf(buf, sizeof(buf), "Redirected Output");
      cv::putText(outputFrame, buf, cv::Point(80, 40), cv::FONT_HERSHEY_SIMPLEX, fontsize, cv::Scalar(0, 255, 0), 1);
    }
  }
    if (FLAG_offlineMode) {
      if (splitScreenView && FLAG_gazeRedirect) {
        // Store the original and redirected image side-by-side
        cv::Mat matDst(cv::Size(outputFrame.cols * 2, outputFrame.rows), outputFrame.type(), cv::Scalar::all(0));
        cv::Mat matRoi = matDst(cv::Rect(0, 0, outputFrame.cols, outputFrame.rows));
        frame.copyTo(matRoi);
        matRoi = matDst(cv::Rect(outputFrame.cols, 0, outputFrame.cols, outputFrame.rows));
        outputFrame.copyTo(matRoi);
        gazeRedirectOutputVideo.write(matDst);
      } else {
        if (!FLAG_gazeRedirect) {
          gazeRedirectOutputVideo.write(frame);
        } else {
          gazeRedirectOutputVideo.write(outputFrame);
        }  
      }
    }
    frameTimer.resume();
#endif  //VISUALIZE
    // If captureOutputs has been set to true, save the output frames.
    if (FLAG_captureOutputs) {
      // Display original gives the option to display the original frame in addition to the redirected image
      // side-by-side.
      if (splitScreenView && FLAG_gazeRedirect) {
        // Store the original and redirected image side-by-side
        cv::Mat matDst(cv::Size(outputFrame.cols * 2, outputFrame.rows), outputFrame.type(), cv::Scalar::all(0));
        cv::Mat matRoi = matDst(cv::Rect(0, 0, outputFrame.cols, outputFrame.rows));
        frame.copyTo(matRoi);
        matRoi = matDst(cv::Rect(outputFrame.cols, 0, outputFrame.cols, outputFrame.rows));
        outputFrame.copyTo(matRoi);
        writeVideo(matDst);
      } else {
        if (!FLAG_gazeRedirect) {
          writeVideo(frame);
        } else {
          writeVideo(outputFrame);
        }
      }
    }
  return doErr;
}

DoApp::Err DoApp::initCamera(const char *camRes, unsigned int camID) {
  if (cap.open(camID)) {
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

      // openCV API(CAP_PROP_FRAME_WIDTH) to get camera resolution is not always reliable with some cameras
      cap >> frame;
      if (frame.empty()) return errCamera;
      if (inputWidth != frame.cols || inputHeight != frame.rows) {
        std::cout << "!!! warning: openCV API(CAP_PROP_FRAME_WIDTH/CV_CAP_PROP_FRAME_HEIGHT) to get camera resolution is not trustable. Using the resolution from the actual frame" << std::endl;
        inputWidth = frame.cols;
        inputHeight = frame.rows;
      }

      gaze_ar_engine.setInputImageWidth(inputWidth);
      gaze_ar_engine.setInputImageHeight(inputHeight);
    }
  } else
    return errCamera;
  return errNone;
}

DoApp::Err DoApp::initOfflineMode(const char *inputFilename, const char *outputFilename) {
  if (cap.open(inputFilename)) {
    inputWidth = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    inputHeight = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    gaze_ar_engine.setInputImageWidth(inputWidth);
    gaze_ar_engine.setInputImageHeight(inputHeight);
  } else {
    printf("ERROR: Unable to open the input video file \"%s\" \n", inputFilename);
    return Err::errVideo;
  }
  std::string fdOutputVideoName, fldOutputVideoName, ffOutputVideoName, fgzOutputVideoName;
  std::string outputFilePrefix;
  if (outputFilename && strlen(outputFilename) != 0) {
    outputFilePrefix = outputFilename;
  } else {
    size_t lastindex = std::string(inputFilename).find_last_of(".");
    outputFilePrefix = std::string(inputFilename).substr(0, lastindex);
    outputFilePrefix = outputFilePrefix + "_gaze.mp4";
  }
  fgzOutputVideoName = outputFilePrefix;
  if (splitScreenView && FLAG_gazeRedirect) {
    if (!gazeRedirectOutputVideo.open(fgzOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                      cv::Size(inputWidth * 2, inputHeight))) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", fgzOutputVideoName.c_str());
      return Err::errGeneral;
    }
  } else {
    if (!gazeRedirectOutputVideo.open(fgzOutputVideoName, StringToFourcc(FLAG_captureCodec), cap.get(CV_CAP_PROP_FPS),
                                      cv::Size(inputWidth, inputHeight))) {
      printf("ERROR: Unable to open the output video file \"%s\" \n", fgzOutputVideoName.c_str());
      return Err::errGeneral;
    }
  }
  captureVideo = true;
  return Err::errNone;
}

DoApp::DoApp() {
  // Make sure things are initialized properly
  gApp = this;
  drawVisualization = FLAG_drawVisualization;
  showFPS = false;
  splitScreenView = FLAG_splitScreenView;
  displayLandmarks = true;
  captureVideo = false;
  frameTime = 0;
  frameIndex = 0;
  nvErr = GazeEngine::errNone;
}

DoApp::~DoApp() {}

char *g_nvARSDKPath = NULL;

int chooseGPU() {
  // If the system has multiple supported GPUs then the application
  // should use CUDA driver APIs or CUDA runtime APIs to enumerate
  // the GPUs and select one based on the application's requirements

  // Cuda device 0
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
  snprintf(buf, sizeof(buf), "Kalman %s", (gaze_ar_engine.bStabilizeFace ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

void DoApp::drawVideoCaptureStatus(cv::Mat &img) {
  char buf[32];
  snprintf(buf, sizeof(buf), "Video Capturing %s", (captureVideo ? "on" : "off"));
  cv::putText(img, buf, cv::Point(10, img.rows - 70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
}

DoApp::Err DoApp::run() {
  DoApp::Err doErr = errNone;

  GazeEngine::Err err = gaze_ar_engine.initGazeRedirectionIOParams();
  if (err != GazeEngine::Err::errNone) {
    return doAppErr(err);
  }

  while (1) {
    doErr = acquireFrame();
    if (frame.empty() && FLAG_offlineMode) {
      // We have reached the end of the video
      // so return without any error.
      return DoApp::errNone;
    } else if (doErr != DoApp::errNone) {
      return doErr;
    }
    outputFrame.create(inputHeight, inputWidth, frame.type());
    doErr = acquireGazeRedirection();
    if (DoApp::errCancel == doErr || DoApp::errVideo == doErr) return doErr;
#ifdef VISUALIZE
    if (!frame.empty() && !FLAG_offlineMode) {
      if (drawVisualization) {
        drawFPS(frame);
        drawKalmanStatus(frame);
        if (FLAG_captureOutputs && captureVideo) {
          if (!FLAG_gazeRedirect) {
            drawVideoCaptureStatus(frame);
          } else {
            drawVideoCaptureStatus(outputFrame);
          }
          
        }
      }
      if (splitScreenView && FLAG_gazeRedirect) {
        // Store the original and redirected image side-by-side
        cv::Mat matDst(cv::Size(outputFrame.cols * 2, outputFrame.rows), outputFrame.type(), cv::Scalar::all(0));
        cv::Mat matRoi = matDst(cv::Rect(0, 0, outputFrame.cols, outputFrame.rows));
        frame.copyTo(matRoi);
        matRoi = matDst(cv::Rect(outputFrame.cols, 0, outputFrame.cols, outputFrame.rows));
        outputFrame.copyTo(matRoi);
        cv::imshow(windowTitle, matDst);
      } else {
        if (!FLAG_gazeRedirect) {
          cv::imshow(windowTitle, frame);
        } else {
          cv::imshow(windowTitle, outputFrame);
        }
      }
    }
#endif  //VISUALIZE
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
    {errInitialization, "Initializing Gaze Engine failed"},
    {errRead, "an error has occured while reading a file"},
    {errEffect, "an error has occured while creating a feature"},
    {errParameter, "an error has occured while setting a parameter for a feature"},
    {errUnimplemented, "the feature is unimplemented"},
    {errMissing, "missing input parameter"},
    {errVideo, "no video source has been found"},
    {errImageSize, "the image size cannot be accommodated"},
    {errNotFound, "the item cannot be found"},
    {errGLFWInit, "GLFW initialization failed"},
    {errGLInit, "OpenGL initialization failed"},
    {errRendererInit, "renderer initialization failed"},
    {errGLResource, "an OpenGL resource could not be found"},
    {errGLGeneric, "an otherwise unspecified OpenGL error has occurred"},
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
  if (FLAG_verbose) printf("Enable temporal optimizations in detecting face and landmarks = %d\n", FLAG_temporal);
  app.gaze_ar_engine.setFaceStabilization(FLAG_temporal);

  if (FLAG_offlineMode) {
    if (FLAG_inFile.empty()) {
      doErr = DoApp::errMissing;
      printf("ERROR: %s, please specify input file using --in \n", app.errorStringFromCode(doErr));
      goto bail;
    }
    doErr = app.initOfflineMode(FLAG_inFile.c_str(), FLAG_outFile.c_str());
  } else {
    doErr = app.initCamera(FLAG_camRes.c_str(), FLAG_camID);
  }
  BAIL_IF_ERR(doErr);

  doErr = app.initGazeEngine(FLAG_modelPath.c_str(), FLAG_isNumLandmarks126, FLAG_gazeRedirect, FLAG_eyeSizeSensitivity, FLAG_useCudaGraph);
  BAIL_IF_ERR(doErr);

  doErr = app.run();
  BAIL_IF_ERR(doErr);

bail:
  if (doErr) printf("ERROR: %s\n", app.errorStringFromCode(doErr));
  app.stop();
  return (int)doErr;
}

