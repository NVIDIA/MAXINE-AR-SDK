/*###############################################################################
#
# Copyright 2021 NVIDIA Corporation
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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"
#include "MeshRenderer.h"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif // WIN32_LEAN_AND_MEAN
  #include <Windows.h>
  #define strcasecmp _stricmp
#else
  #include <sys/stat.h>
#endif // _WIN32

#ifdef _ENABLE_UI
#include "ExpressionAppUI.h"
#endif  // _ENABLE_UI

#if CV_MAJOR_VERSION >= 4
  #define CV_CAP_PROP_FPS           cv::CAP_PROP_FPS
  #define CV_CAP_PROP_FRAME_COUNT   cv::CAP_PROP_FRAME_COUNT
  #define CV_CAP_PROP_FRAME_HEIGHT  cv::CAP_PROP_FRAME_HEIGHT
  #define CV_CAP_PROP_FRAME_WIDTH   cv::CAP_PROP_FRAME_WIDTH
  #define CV_CAP_PROP_POS_FRAMES    cv::CAP_PROP_POS_FRAMES
  #define CV_INTER_AREA             cv::INTER_AREA
  #define CV_INTER_LINEAR           cv::INTER_LINEAR
#endif // CV_MAJOR_VERSION

#ifndef M_PI
  #define M_PI                      3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
  #define M_2PI                     6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
  #define M_PI_2                    1.5707963267948966192
#endif /* M_PI_2 */
#define D_RADIANS_PER_DEGREE        (M_PI / 180.)
#define F_PI                        ((float)M_PI)
#define F_PI_2                      ((float)M_PI_2)
#define F_2PI                       ((float)M_2PI)
#define F_RADIANS_PER_DEGREE        (float)(M_PI / 180.)
#define CTL(x)                      ((x) & 0x1F)
#define HELP_REQUESTED              411

#define BAIL_IF_ERR(err)            do { if ((int)(err) != 0)     { goto bail;             } } while(0)
#define BAIL_IF_NULL(x, err, code)  do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)
#define BAIL_IF_CUERR(cu, err)      do { if (cudaSuccess != (cu)) { err = NvFromCuErr(cu); } } while(0)
#define BAIL(err, code)             do {                            err = code; goto bail;   } while(0)

#define DEFAULT_CODEC         "avc1"
#define DEFAULT_FACE_MODEL    "face_model2.nvf"
#define DEFAULT_RENDER_MODEL  "face_model2.nvf"
#define NUM_CAMERA_INTRINSIC_PARAMS 3

/********************************************************************************
 * Command-line arguments
 ********************************************************************************/

bool
    FLAG_debug              = false,
    FLAG_loop               = false,
    FLAG_show               = false,
    FLAG_showUI             = false,
    FLAG_verbose            = false;
std::string
    FLAG_camRes,
    FLAG_codec              = DEFAULT_CODEC,
    FLAG_fitModel           = DEFAULT_FACE_MODEL,
    FLAG_inFile,
    FLAG_modelDir,
    FLAG_outDir,
    FLAG_outFile,
    FLAG_renderModel        = DEFAULT_RENDER_MODEL;
int
    FLAG_filter             = NVAR_TEMPORAL_FILTER_FACE_BOX
                            | NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS
                            | NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE
                            | NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS
                            | NVAR_TEMPORAL_FILTER_FACIAL_GAZE,
                            //| NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS,
    FLAG_viewMode           = 0xF,                // VIEW_MESH | VIEW_IMAGE | VIEW_PLOT | VIEW_LM
    FLAG_exprMode           = 2,                  // 1=mesh, 2=MLP
    FLAG_poseMode           = 0,
    FLAG_cheekPuff          = 0,
    FLAG_gaze               = 0;
double
    FLAG_fov                = 0.0;                // Orthographic by default


/********************************************************************************
 * Usage
 ********************************************************************************/

static void Usage() {
  printf(
      "ExpressionApp [<args> ...]\n"
      "where <args> is\n"
      " --cam_res=[WWWx]HHH         specify resolution as height or width x height\n"
      " --codec=<fourcc>            FOURCC code for the desired codec (default avc1)\n"
      " --debug[=(true|false)]      report debugging info (default false)\n"
      " --expr_mode=<number>        SDK feature used for generation expressions: 1=Face3DReconstruction, 2=FaceExpressions (default 2)\n"
      " --pose_mode=<number>        Pose mode used for the FaceExpressions feature only: 0:3DOF, 1:6DOF (default 0)\n"
      " --cheekpuff[=(true|false)]  (Experimental) Enable cheek puff blendshapes (default=false)\n"
      " --face_model=<file>         specify the  face model to be used for fitting (default " DEFAULT_FACE_MODEL ")\n"
      " --filter=<bitfield>         1: face box, 2: landmarks, 4: pose, 16: expressions, 32: gaze,\n"
      "                             256: eye and mouth closure\n"
      "                             (default 55: face box, landmarks, pose, expressions, gaze; no closure)\n"
      " --gaze=<number>             specify gaze estimation mode 0=implicit, 1=explicit (default 0)\n"
      " --fov=<degrees>             field of view, in degrees; 0 implies orthographic (default 0)\n"
      " --help                      print this message\n"
      " --in=<file>                 specify the input file (default webcam 0)\n"
      " --loop[=(true|false)]       play the same video repeatedly\n"
      " --model_dir=<path>          specify the directory containing the TRT models\n"
      " --model_path=<path>         specify the directory containing the TRT models\n"
      " --out=<file>                specify the output file\n"
      " --render_model=<file>       specify the face model to be used for rendering (default " DEFAULT_RENDER_MODEL ")\n"
      " --show[=(true|false)]       show the results (default false, unless --out is empty)\n"
      " --show_ui[=(true|false)]    show the expression calibration UI (default false)\n"
      " --temporal=<bitfield>       apply temporal filter: see --filter\n"
      " --view_mode=<bitfield>      1: mesh, 2: image, 4: plot, 8: landmarks (default 15: all)\n"
      " --verbose[=(true|false)]    report interesting info (default off)\n"
      "Keyboard commands:\n"
      " escape      - quit\n"
      " q or Q      - quit\n"
      " m           - toggle mesh display\n"
      " n           - calibrate expression weights\n"
      " i           - toggle image display\n"
      " p           - toggle plot display\n"
      " l           - toggle landmark display\n"
      " f           - toggle frame rate display\n"
      " L or ctrl-L - toggle landmark filtering\n"
      " N or ctrl-N - un-calibrate expression weights\n"
      " P or ctrl-P - toggle pose filtering\n"
      " E or ctrl-E - toggle expression filtering\n"
      " G or ctrl-G - toggle gaze filtering\n"
      " C or ctrl-C - toggle closure enhancement\n"
      " 1           - expressions from mesh fitting\n"
      " 2           - expressions from DNN\n"
  );
}

static bool GetFlagArgVal(const char *flag, const char *arg, const char **val) {
  if (*arg != '-')
    return false;
  while (*++arg == '-')
    continue;
  const char *s = strchr(arg, '=');
  if (s == nullptr) {
    if (strcmp(flag, arg))
      return false;
    *val = nullptr;
    return true;
  }
  unsigned n = (unsigned)(s - arg);
  if ((strlen(flag) != n) || (strncmp(flag, arg, n) != 0))
    return false;
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
  if (success)
    *val = (!valStr                   ||
          !strcasecmp(valStr, "true") ||
          !strcasecmp(valStr, "on")   ||
          !strcasecmp(valStr, "yes")  ||
          !strcasecmp(valStr, "1"));
  return success;
}

bool GetFlagArgVal(const char *flag, const char *arg, long *val) {
  const char *valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success)
    *val = strtol(valStr, nullptr, 0);
  return success;
}

static bool GetFlagArgVal(const char *flag, const char *arg, int *val) {
  long longVal;
  bool success = GetFlagArgVal(flag, arg, &longVal);
  if (success)
    *val = (int)longVal;
  return success;
}

bool GetFlagArgVal(const char* flag, const char* arg, double* val) {
  const char* valStr;
  bool success = GetFlagArgVal(flag, arg, &valStr);
  if (success)
    *val = valStr ? strtod(valStr, nullptr) : 1.0;
  return success;
}


/********************************************************************************
 * StringToFourcc
 ********************************************************************************/

static int StringToFourcc(const std::string &str) {
  union chint { int i; char c[4]; };
  chint x = {0};
  for (int n = (str.size() < 4) ? (int)str.size() : 4; n--;)
    x.c[n] = str[n];
  return x.i;
}


/********************************************************************************
 * ParseMyArgs
 ********************************************************************************/

static int ParseMyArgs(int argc, char **argv) {
  // query NVAR_MODEL_DIR environment variable first before checking the command line arguments
  const char* modelPath = getenv("NVAR_MODEL_DIR");
  if (modelPath) {
    FLAG_modelDir = modelPath;
  }

  int errs = 0;
  for (--argc, ++argv; argc--; ++argv) {
    bool help;
    const char *arg = *argv;
    if (arg[0] != '-') {
      continue;
    } else if ((arg[1] == '-') && (
      GetFlagArgVal("cam_res",      arg, &FLAG_camRes)        || 
      GetFlagArgVal("codec",        arg, &FLAG_codec)         ||
      GetFlagArgVal("debug",        arg, &FLAG_debug)         ||
      GetFlagArgVal("expr_mode",    arg, &FLAG_exprMode)      ||
      GetFlagArgVal("pose_mode",    arg, &FLAG_poseMode)      ||
      GetFlagArgVal("cheekpuff",    arg, &FLAG_cheekPuff)     ||
      GetFlagArgVal("face_model",   arg, &FLAG_fitModel)      ||
      GetFlagArgVal("filter",       arg, &FLAG_filter)        ||
      GetFlagArgVal("gaze",         arg, &FLAG_gaze)          ||
      GetFlagArgVal("fov",          arg, &FLAG_fov)           ||
      GetFlagArgVal("in",           arg, &FLAG_inFile)        ||
      GetFlagArgVal("in_file",      arg, &FLAG_inFile)        ||
      GetFlagArgVal("loop",         arg, &FLAG_loop)          ||
      GetFlagArgVal("model_dir",    arg, &FLAG_modelDir)      ||
      GetFlagArgVal("model_path",   arg, &FLAG_modelDir)      ||
      GetFlagArgVal("out",          arg, &FLAG_outFile)       ||
      GetFlagArgVal("out_file",     arg, &FLAG_outFile)       ||
      GetFlagArgVal("render_model", arg, &FLAG_renderModel)   ||
      GetFlagArgVal("show",         arg, &FLAG_show)          ||
      GetFlagArgVal("show_ui",      arg, &FLAG_showUI)        ||
      GetFlagArgVal("temporal",     arg, &FLAG_filter)        ||
      GetFlagArgVal("verbose",      arg, &FLAG_verbose)       ||
      GetFlagArgVal("view_mode",    arg, &FLAG_viewMode)
    )) {
      continue;
    } else if (GetFlagArgVal("help", arg, &help)) {
      Usage();
      errs = HELP_REQUESTED;
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


static bool FileExists(const char* fileName) {
  #ifdef _MSC_VER
    DWORD attributes = GetFileAttributesA(fileName);
    return attributes != INVALID_FILE_ATTRIBUTES;
  #else  // !MSC_VER
    struct stat statBuf;
    return 0 == stat(fileName, &statBuf);
  #endif  // MSC_VER
}
inline bool FileExists(const std::string& str) { return FileExists(str.c_str()); }
static bool SetDirIfFileExists(const std::string& testDir, const std::string& file, std::string& resultDir) {
  std::string dirFile = (testDir + '/' + file);
  bool exists = FileExists(dirFile);
  if (exists)
    resultDir = testDir;
  return exists;
}
static bool SetPathIfFileExists(const std::string& testDir, const std::string& file, std::string& path) {
  std::string dirFile = testDir.empty() ? file : (testDir + '/' + file);
  bool exists = FileExists(dirFile);
  if (exists)
    path = dirFile;
  return exists;
}

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

inline NvCV_Status NvFromAppErr(int appErr) { return (NvCV_Status)appErr; }


class App {
public:

  App() {}
  ~App() { stop(); }

  NvCV_Status run();
  NvCV_Status stop();
  NvCV_Status setInputVideo(const std::string& file); // open immediately
  NvCV_Status setInputCamera(int index, const std::string& resStr); // open immediately
  NvCV_Status setOutputVideo(const std::string& file);
  NvCV_Status set(int codec, double fps, unsigned width, unsigned height);  // deferred open
  NvCV_Status init();
  NvCV_Status resizeDst();
  NvCV_Status openOutputVideo(int codec, double fps, unsigned width, unsigned height);
  NvCV_Status initFaceFit();
  NvCV_Status initMLPExpressions();
  NvCV_Status calibrateExpressionWeights();
  NvCV_Status unCalibrateExpressionWeights();
  NvCV_Status normalizeExpressionsWeights();
  NvCV_Status updateCamera();
  NvCV_Status toggleFaceBoxFiltering();
  NvCV_Status toggleLandmarkFiltering();
  NvCV_Status togglePoseFiltering();
  NvCV_Status toggleExpressionFiltering();
  NvCV_Status toggleGazeFiltering();
  NvCV_Status toggleClosureEnhancement();
  NvCV_Status togglePoseMode();
  NvCV_Status overlayLandmarks(const float landmarks[126 * 2], unsigned screenHeight, NvCVImage *im);
  void getFPS();
  void drawFPS(cv::Mat& img);
  void barPlotExprs();

  const char *getErrorStringFromCode(NvCV_Status err);

  struct Pose {
    NvAR_Quaternion rotation;
    NvAR_Vector3f translation;
    float* data() { return &rotation.x; }
    const float* data() const { return &rotation.x; }
  };
  CUstream          _stream = 0;
  cv::Mat           _ocvSrcImg, _ocvDstImg; // _ocvSrcImg is allocated, _ocvDstImg is just a wrapper
  cv::VideoCapture  _vidIn{};
  cv::VideoWriter   _vidOut{};
  double            _frameRate;
  int               _miniX, _miniY, _renderX, _renderY, _plotX, _plotY;
  NvAR_FaceMesh     _arMesh { nullptr, 0, nullptr, 0 };
  NvAR_FeatureHandle _featureHan{};
  NvAR_RenderingParams _renderParams;
  NvCVImage         _srcImg, _compImg, _srcGpu, _renderImg; // wrapper, alloced, alloced, alloced
  Pose              _pose;
  float             _cameraIntrinsicParams[NUM_CAMERA_INTRINSIC_PARAMS];
  std::string       _inFile, _outFile;
  std::vector<NvAR_Rect> _outputBboxData;
  NvAR_BBoxes        _outputBboxes;
  std::vector<float> _expressions, _expressionZeroPoint, _expressionScale, _expressionExponent, _eigenvalues,
                     _landmarkConfidence;
  std::vector<NvAR_Point2f> _landmarks;
  std::vector<NvAR_Vector3f> _vertices;
  std::vector< NvAR_Vector3u16> _triangles;
  unsigned          _videoWidth, _videoHeight, _miniWidth, _miniHeight, _renderWidth, _renderHeight,
                    _plotWidth, _plotHeight, _compWidth, _compHeight, _eigenCount, _exprCount, _landmarkCount,
                    _viewMode, _exprMode, _filtering;
  unsigned          _poseMode = 0;
  bool              _enableCheekPuff;
  MeshRendererBroker _broker;
  MeshRenderer      *_renderer = nullptr;
  static const char _windowTitle[], *_exprAbbr[][4], *_sfmExprAbbr[][4];
  MyTimer           _timer;
  bool              _showFPS;
  bool              _performCalibration;
  bool              _cameraNeedsUpdate;
  double            _frameTime;
  float             _globalExpressionParam;
#ifdef _ENABLE_UI
  ExpressionAppUI   ui_obj_;
#endif  // _ENABLE_UI
  enum {
    EXPR_MODE_MESH  = 1,
    EXPR_MODE_MLP   = 2,
  };
  enum {
    VIEW_MESH   = (1 << 0),
    VIEW_IMAGE  = (1 << 1),
    VIEW_PLOT   = (1 << 2),
    VIEW_LM     = (1 << 3)
  };
  enum {
    APP_ERR_GENERAL = 1,
    APP_ERR_OPEN,
    APP_ERR_READ,
    APP_ERR_WRITE,
    APP_ERR_INIT,
    APP_ERR_RUN,
    APP_ERR_EFFECT,
    APP_ERR_PARAM,
    APP_ERR_UNIMPLEMENTED,
    APP_ERR_MISSING,
    APP_ERR_VIDEO,
    APP_ERR_IMAGE_SIZE,
    APP_ERR_NOT_FOUND,
    APP_ERR_FACE_MODEL,
    APP_ERR_GLFW_INIT,
    APP_ERR_GL_INIT,
    APP_ERR_RENDER_INIT,
    APP_ERR_GL_RESOURCE,
    APP_ERR_GL_GENERAL,
    APP_ERR_FACE_FIT,
    APP_ERR_NO_FACE,
    APP_ERR_CANCEL,
    APP_ERR_CAMERA,
    APP_ERR_ARG_PARSE,
    APP_ERR_EOF
  };
};
const char App::_windowTitle[] = "Expression App";
const char *App::_exprAbbr[][4] = {
  { "BROW", "DOWN", "LEFT", NULL    },    // 0  browDown_L
  { "BROW", "DOWN", "RIGHT",NULL    },    // 1  browDown_R
  { "BROW", "INNR", "UP",   "LEFT"  },    // 2  browInnerUp_L
  { "BROW", "INNR", "UP",   "RIGHT" },    // 3  browInnerUp_R
  { "BROW", "OUTR", "UP",   "LEFT"  },    // 4  browOuterUp_L
  { "BROW", "OUTR", "UP",   "RIGHT" },    // 5  browOuterUp_R
  { "CHEE", "PUFF", "LEFT", NULL    },    // 6  cheekPuff_L
  { "CHEE", "PUFF", "RIGHT",NULL    },    // 7  cheekPuff_R
  { "CHEE", "SQNT", "LEFT", NULL    },    // 8  cheekSquint_L
  { "CHEE", "SQNT", "RIGHT",NULL    },    // 9  cheekSquint_R
  { "EYE",  "BLNK", "LEFT", NULL    },    // 10 eyeBlink_L
  { "EYE",  "BLNK", "RIGHT",NULL    },    // 11 eyeBlink_R
  { "EYE",  "LOOK", "DOWN", "LEFT"  },    // 12 eyeLookDown_L
  { "EYE",  "LOOK", "DOWN", "RIGHT" },    // 13 eyeLookDown_R
  { "EYE",  "LOOK", "IN",   "LEFT"  },    // 14 eyeLookIn_L
  { "EYE",  "LOOK", "IN",   "RIGHT" },    // 15 eyeLookIn_R
  { "EYE",  "LOOK", "OUT",  "LEFT"  },    // 16 eyeLookOut_L
  { "EYE",  "LOOK", "OUT",  "RIGHT" },    // 17 eyeLookOut_R
  { "EYE",  "LOOK", "UP",   "LEFT"  },    // 18 eyeLookUp_L
  { "EYE",  "LOOK", "UP",   "RIGHT" },    // 19 eyeLookUp_R
  { "EYE",  "SQNT", "LEFT", NULL    },    // 20 eyeSquint_L
  { "EYE",  "SQNT", "RIGHT",NULL    },    // 21 eyeSquint_R
  { "EYE",  "WIDE", "LEFT", NULL    },    // 22 eyeWide_L
  { "EYE",  "WIDE", "RIGHT",NULL    },    // 23 eyeWide_R
  { "JAW",  "FWD",  NULL,   NULL    },    // 24 jawForward
  { "JAW",  "LEFT", NULL,   NULL    },    // 25 jawLeft
  { "JAW",  "OPEN", NULL,   NULL    },    // 26 jawOpen
  { "JAW",  "RIGHT",NULL   ,NULL    },    // 27 jawRight
  { "MOUT", "CLOS", NULL,   NULL    },    // 28 mouthClose
  { "MOUT", "DMPL", "LEFT", NULL    },    // 29 mouthDimple_L
  { "MOUT", "DMPL", "RIGHT",NULL    },    // 30 mouthDimple_R
  { "MOUT", "FRWN", "LEFT", NULL    },    // 31 mouthFrown_L
  { "MOUT", "FRWN", "RIGHT",NULL    },    // 32 mouthFrown_R
  { "MOUT", "FUNL", NULL,   NULL    },    // 33 mouthFunnel
  { "MOUT", "LEFT", NULL,   NULL    },    // 34 mouthLeft
  { "MOUT", "LOWR", "DOWN", "LEFT"  },    // 35 mouthLowerDown_L
  { "MOUT", "LOWR", "DOWN", "RIGHT" },    // 36 mouthLowerDown_R
  { "MOUT", "PRES", "LEFT", NULL    },    // 37 mouthPress_L
  { "MOUT", "PRES", "RIGHT",NULL    },    // 38 mouthPress_R
  { "MOUT", "PUKR", NULL,   NULL    },    // 39 mouthPucker
  { "MOUT", "RIGHT",NULL,   NULL    },    // 40 mouthRight
  { "MOUT", "ROLL", "LOWR", NULL    },    // 41 mouthRollLower
  { "MOUT", "ROLL", "UPPR", NULL    },    // 41 mouthRollUpper
  { "MOUT", "SHRG", "LOWR", NULL    },    // 43 mouthShrugLower
  { "MOUT", "SHRG", "UPPR", NULL    },    // 44 mouthShrugUpper
  { "MOUT", "SMIL", "LEFT", NULL    },    // 45 mouthSmile_L
  { "MOUT", "SMIL", "RIGHT",NULL    },    // 46 mouthSmile_R
  { "MOUT", "STRH", "LEFT", NULL    },    // 47 mouthStretch_L
  { "MOUT", "STRH", "RIGHT",NULL    },    // 48 mouthStretch_R
  { "MOUT", "UPPR", "UP",   "LEFT"  },    // 49 mouthUpperUp_L
  { "MOUT", "UPPR", "UP",   "RIGHT" },    // 50 mouthUpperUp_R
  { "NOSE", "SNER", "LEFT", NULL    },    // 51 noseSneer_L
  { "NOSE", "SNER", "RIGHT",NULL    },    // 52 noseSneer_R
};
const char *App::_sfmExprAbbr[][4] = {
  { "ANGER",    NULL, NULL, NULL    },
  { "DISGUST",  NULL, NULL, NULL    },
  { "FEAR",     NULL, NULL, NULL    },
  { "HAPPY",    NULL, NULL, NULL    },
  { "SAD",      NULL, NULL, NULL    },
  { "SURPRISE", NULL, NULL, NULL    },
};

NvCV_Status App::setInputVideo(const std::string& file) {
  // opencv2/vidioio.hpp
  if (!_vidIn.open(file)) return (NvCV_Status)APP_ERR_OPEN;
  _frameRate   = _vidIn.get(CV_CAP_PROP_FPS);
  _videoWidth  = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_WIDTH);
  _videoHeight = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_HEIGHT);
  if (FLAG_verbose)
    printf("Video capture resolution set to %dx%d @ %4.1f fps\n", _videoWidth, _videoHeight, _frameRate);
  _inFile = file;
  return NVCV_SUCCESS;
}

NvCV_Status App::setInputCamera(int index, const std::string& resStr) {
  if (!_vidIn.open(index))
    return NvFromAppErr(APP_ERR_OPEN);
  if (!resStr.empty()) {
    int n, width, height;
    n = sscanf(resStr.c_str(), "%d%*[xX]%d", &width, &height);
    switch (n) {
      case 2:
        break;  // We have read both width and height
      case 1:
        height = width;
        width = (int)(height * (4. / 3.) + .5);
        break;
      default:
        height = 0;
        width = 0;
        break;
    }
    if (width)  _vidIn.set(CV_CAP_PROP_FRAME_WIDTH,  width);
    if (height) _vidIn.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    _inFile = "webcam";
    _inFile += std::to_string(index);
  }
  _videoWidth  = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_WIDTH);
  _videoHeight = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_HEIGHT);
  _frameRate = _vidIn.get(CV_CAP_PROP_FPS);
  // Rounding the frame rate is required because OpenCV does not support all frame rates when writing video
  static const int fps_precision = 1000;
  _frameRate = static_cast<int>((_frameRate + 0.5) * fps_precision) / static_cast<double>(fps_precision);
  if (FLAG_verbose)
    printf("Camera capture resolution set to %dx%d @ %4.1f fps\n", _videoWidth, _videoHeight, _frameRate);
  return NVCV_SUCCESS;
}

NvCV_Status App::setOutputVideo(const std::string& file) {
  _outFile = file;
  return NVCV_SUCCESS;
}

NvCV_Status App::openOutputVideo(int codec, double fps, unsigned width, unsigned height) {
  cv::Size sz;
  sz.width = width;
  sz.height = height;
  if (_outFile.empty()) {
    if (_inFile.size() <= 4)
      return NvFromAppErr(APP_ERR_OPEN);
    _outFile.assign(_inFile, 0, _inFile.size() - 4);
    _outFile += "_out.mp4";
  }
  _vidOut.open(_outFile, codec, fps, sz, true);
  return _vidOut.isOpened() ? NVCV_SUCCESS : NvFromAppErr(APP_ERR_OPEN);
}

NvCV_Status App::stop() {
#if _ENABLE_UI
  if(FLAG_showUI)
    ui_obj_.cleanup();
#endif  // _ENABLE_UI
  if (_vidOut.isOpened()) _vidOut.release();
  if (_vidIn.isOpened())  _vidIn.release();
  if (_featureHan)        NvAR_Destroy(_featureHan);
  if (_renderer)          _renderer->destroy();
  _renderer = nullptr;
  _featureHan = nullptr;
  _inFile.clear();
  _outFile.clear();
  return NVCV_SUCCESS;
}


const char *App::getErrorStringFromCode(NvCV_Status err) {
  int intErr = (int)err;
  if (intErr > 0) {
    struct LUTEntry { int code; const char *string; };
    static const struct LUTEntry lut[] {
      { APP_ERR_GENERAL,        "General application error" },
      { APP_ERR_READ,           "Read error" },
      { APP_ERR_WRITE,          "Write error" },
      { APP_ERR_INIT,           "Initialization error" },
      { APP_ERR_RUN,            "Run error" },
      { APP_ERR_EFFECT,         "Error creating an effect" },
      { APP_ERR_PARAM,          "Parameter error" },
      { APP_ERR_UNIMPLEMENTED,  "Unimplemented" },
      { APP_ERR_MISSING,        "Something is missing" },
      { APP_ERR_VIDEO,          "Video error" },
      { APP_ERR_IMAGE_SIZE,     "Image size error" },
      { APP_ERR_NOT_FOUND,      "Not found" },
      { APP_ERR_FACE_MODEL,     "Face model error" },
      { APP_ERR_GLFW_INIT,      "Error initializing GLFW" },
      { APP_ERR_GL_INIT,        "Error initializing OpenGL" },
      { APP_ERR_RENDER_INIT,    "Error initializing the renderer" },
      { APP_ERR_GL_RESOURCE,    "OpenGL resource error" },
      { APP_ERR_GL_GENERAL,     "General OpenGL error" },
      { APP_ERR_FACE_FIT,       "Face fit error" },
      { APP_ERR_NO_FACE,        "No face was found" },
      { APP_ERR_CANCEL,         "The operation has been canceled" },
      { APP_ERR_CAMERA,         "Camera error" },
      { APP_ERR_ARG_PARSE,      "Argument parsing error" },
    };
    for (const LUTEntry *p = lut; p != &lut[sizeof(lut) / sizeof(lut[0])]; ++p)
      if (intErr == p->code)
        return p->string;
  }
  return NvCV_GetErrorStringFromCode(err);
}

char *g_nvARSDKPath = NULL;


NvCV_Status ResizeNvCVImage(const NvCVImage *src, NvCVImage *dst) {
  int interpolation = ((double)dst->width * dst->height / (src->width * src->height) < 1.) ?
                      CV_INTER_AREA : CV_INTER_LINEAR;
  cv::Mat ocvSrc, ocvDst;
  CVWrapperForNvCVImage(src, &ocvSrc);
  CVWrapperForNvCVImage(dst, &ocvDst);
  cv::resize(ocvSrc, ocvDst, ocvDst.size(), 0, 0, interpolation);
  return NVCV_SUCCESS;
}


NvCV_Status App::initFaceFit() {
  unsigned modelLandmarks = 126, n;
  NvCV_Status err;

  // Initialize AR effect
  if (_featureHan) {
    if (EXPR_MODE_MESH == _exprMode) return NVCV_SUCCESS;
    NvAR_Destroy(_featureHan);
    _featureHan = nullptr;
  }
  BAIL_IF_ERR(err = NvAR_Create(NvAR_Feature_Face3DReconstruction, &_featureHan));
  if (!FLAG_modelDir.empty())
    BAIL_IF_ERR(err = NvAR_SetString(_featureHan, NvAR_Parameter_Config(ModelDir), FLAG_modelDir.c_str()));
  if (!FLAG_fitModel.empty()) {
    BAIL_IF_ERR(err = NvAR_SetString(_featureHan, NvAR_Parameter_Config(ModelName), FLAG_fitModel.c_str()));
    if (FLAG_fitModel[FLAG_fitModel.size() - 5] == '0') // face model 0 has 68 landmarks, the others have 126
      modelLandmarks = 68;
  }
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Landmarks_Size), modelLandmarks));
  BAIL_IF_ERR(err = NvAR_SetCudaStream(_featureHan, NvAR_Parameter_Config(CUDAStream), _stream));
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering));
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(GazeMode), FLAG_gaze));
  BAIL_IF_ERR(err = NvAR_Load(_featureHan));
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(ShapeEigenValueCount), &_eigenCount));
  _eigenvalues.resize(_eigenCount);
  _outputBboxData.assign(25, { 0.f, 0.f, 0.f, 0.f });
  _outputBboxes.boxes = _outputBboxData.data();
  _outputBboxes.max_boxes = (uint8_t)_outputBboxData.size();
  _outputBboxes.num_boxes = 0;
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(BoundingBoxes), &_outputBboxes, sizeof(NvAR_BBoxes)));
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(Landmarks_Size), &_landmarkCount));
  _landmarks.resize(_landmarkCount);
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Landmarks), _landmarks.data(), sizeof(NvAR_Point2f)));
  _landmarkConfidence.resize(_landmarkCount);
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(LandmarksConfidence), _landmarkConfidence.data(), _landmarkCount));
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(ExpressionCount), &_exprCount));
  _expressions.resize(_exprCount, 0.0f);
  _expressionZeroPoint.resize(_exprCount, 0.0f);
  _expressionScale.resize(_exprCount, 1.0f);
  _expressionExponent.resize(_exprCount, 1.0f);
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount));
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(ShapeEigenValues), _eigenvalues.data(), _eigenCount));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage)));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Pose), &_pose, sizeof(_pose.rotation)));
  // The following are not used, but apparently required
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(VertexCount), &n));
  _vertices.resize(_arMesh.num_vertices = n);
  _arMesh.vertices = &_vertices[0];
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(TriangleCount), &n));
  _triangles.resize(_arMesh.num_triangles = n);
  _arMesh.tvi = &_triangles[0];
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(FaceMesh), &_arMesh, sizeof(_arMesh)));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(RenderingParams), &_renderParams, sizeof(_renderParams)));
  _exprMode = EXPR_MODE_MESH;
bail:
  return err;
}

NvCV_Status App::initMLPExpressions() {
  const unsigned landmarkCount = 126;
  NvCV_Status err;

  // Initialize AR effect
  if (_featureHan) {
    if (EXPR_MODE_MLP == _exprMode) return NVCV_SUCCESS;
    NvAR_Destroy(_featureHan);
    _featureHan = nullptr;
  }
  BAIL_IF_ERR(err = NvAR_Create(NvAR_Feature_FaceExpressions, &_featureHan));
  if (!FLAG_modelDir.empty())
    BAIL_IF_ERR(err = NvAR_SetString(_featureHan, NvAR_Parameter_Config(ModelDir), FLAG_modelDir.c_str()));
  BAIL_IF_ERR(err = NvAR_SetCudaStream(_featureHan, NvAR_Parameter_Config(CUDAStream), _stream));
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering));
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(PoseMode), _poseMode));
  BAIL_IF_ERR(err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(EnableCheekPuff), _enableCheekPuff));
  BAIL_IF_ERR(err = NvAR_Load(_featureHan));
  _outputBboxData.assign(25, { 0.f, 0.f, 0.f, 0.f });
  _outputBboxes.boxes = _outputBboxData.data();
  _outputBboxes.max_boxes = (uint8_t)_outputBboxData.size();
  _outputBboxes.num_boxes = 0;
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(BoundingBoxes), &_outputBboxes, sizeof(NvAR_BBoxes)));
  _landmarks.resize(landmarkCount);
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Landmarks), _landmarks.data(), sizeof(NvAR_Point2f)));
  _landmarkConfidence.resize(landmarkCount);
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(LandmarksConfidence), _landmarkConfidence.data(), landmarkCount));
  BAIL_IF_ERR(err = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(ExpressionCount), &_exprCount));
  _expressions.resize(_exprCount);
  _expressionZeroPoint.resize(_exprCount, 0.0f);
  _expressionScale.resize(_exprCount, 1.0f);
  _expressionExponent.resize(_exprCount, 1.0f);
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage)));
  // Heuristic for focal length if it is not known
  _cameraIntrinsicParams[0] = static_cast<float>(_srcGpu.height);         // focal length
  _cameraIntrinsicParams[1] = static_cast<float>(_srcGpu.width) / 2.0f;   // cx
  _cameraIntrinsicParams[2] = static_cast<float>(_srcGpu.height) / 2.0f;  // cy
  BAIL_IF_ERR(err = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Input(CameraIntrinsicParams), _cameraIntrinsicParams,
                                     NUM_CAMERA_INTRINSIC_PARAMS));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Pose), &_pose.rotation, sizeof(NvAR_Quaternion)));
  BAIL_IF_ERR(err = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(PoseTranslation), &_pose.translation, sizeof(NvAR_Vector3f)));

  _exprMode = EXPR_MODE_MLP;

bail:
  return err;
}

NvCV_Status App::calibrateExpressionWeights() {
  assert(_expressions.size() == _exprCount);
  assert(_expressionScale.size() == _exprCount);
  assert(_expressionZeroPoint.size() == _exprCount);
  _expressionZeroPoint = _expressions;
  for (size_t i = 0; i < _exprCount; i++) {
    _expressionScale[i] = 1.0f / (1.0f - _expressionZeroPoint[i]);
  }
  _performCalibration = false;
  return NVCV_SUCCESS;
}

NvCV_Status App::unCalibrateExpressionWeights() {
  std::fill(_expressionZeroPoint.begin(), _expressionZeroPoint.end(), 0.0f);
  std::fill(_expressionScale.begin(), _expressionScale.end(), 1.0f);
  std::fill(_expressionExponent.begin(), _expressionExponent.end(), 1.0f);
  return NVCV_SUCCESS;
}

NvCV_Status App::normalizeExpressionsWeights() {
  assert(_expressions.size() == _exprCount);
  assert(_expressionScale.size() == _exprCount);
  assert(_expressionZeroPoint.size() == _exprCount);
  for (size_t i = 0; i < _exprCount; i++)  {
    float tempExpr = _expressions[i];
    _expressions[i] = 1.0f - (std::pow(
        1.0f - (std::max(_expressions[i] - _expressionZeroPoint[i], 0.0f) * _expressionScale[i]),
        _expressionExponent[i]));
    _expressions[i] = _globalExpressionParam * _expressions[i] + (1.0f - _globalExpressionParam) * tempExpr;
  }
  return NVCV_SUCCESS;
}

NvCV_Status App::updateCamera() {
  NvCV_Status err;
  if (_poseMode == 1) {
    const NvAR_Vector3f cam_loc = {0, 0, 0};
    const NvAR_Vector3f cam_dir = {0, 0, -1};
    const NvAR_Vector3f cam_up = {0, 1, 0};
    const float focal_length = _cameraIntrinsicParams[0];
    const float v_fov = focal_length == 0.0f ? 0.0f : 2.0f * atan(_videoHeight / (2 * focal_length));
    err = _renderer->setCamera(&cam_loc.vec[0], &cam_dir.vec[0], &cam_up.vec[0], v_fov, 0.01f, 1000.0f);
  } else {
    err = _renderer->setCamera(nullptr, nullptr, nullptr, 0.0f);
  }
  _cameraNeedsUpdate = false;
  return err;
}

NvCV_Status App::init() {
  NvCV_Status err;
  std::string path;
  std::vector<std::string> rendererList;

  _renderHeight = 480;
  _renderWidth  = 480;
  _miniHeight   = _renderHeight;
  _miniWidth    = (unsigned)((float)_videoWidth * _renderHeight / (_videoHeight * 2) + 0.5f) * 2;
  _compWidth    = _miniWidth + _renderWidth;
  _plotWidth    = _compWidth;
  _plotHeight   = 72;
  _compHeight   = _renderHeight + _plotHeight;
  _miniX        = 0;
  _miniY        = 0;
  _renderX      = _miniWidth;
  _renderY      = 0;
  _plotX        = 0;
  _plotY        = _renderHeight;
  _viewMode     = FLAG_viewMode;
  _showFPS      = false;
  _performCalibration = false;
  _frameTime    = 0.f;
  _exprMode     = 0;
  _poseMode     = FLAG_poseMode;
  _enableCheekPuff = FLAG_cheekPuff;
  _featureHan   = nullptr;
  _filtering    = FLAG_filter;
  _globalExpressionParam  = 1.0f;
  _cameraIntrinsicParams[0] = 0.0f;
  _cameraIntrinsicParams[1] = 0.0f;
  _cameraIntrinsicParams[2] = 0.0f;
  _cameraNeedsUpdate = true;

  err = _broker.getMeshRendererList(rendererList);
  if (NVCV_SUCCESS != err) {
    printf("Cannot engage renderer broker: %s\n", NvCV_GetErrorStringFromCode(err));
    return err;
  }
  if (FLAG_verbose) {
    printf("Renderer list:\n");
    for (const std::string& str : rendererList)
      printf("  %s\n", str.c_str());
  }
  if (rendererList.size() < 1) {
    printf("No renderers available to the broker\n");
    return NVCV_ERR_FEATURENOTFOUND;
  }
  err = _broker.create(rendererList[0].c_str(), &_renderer);
  if (NVCV_SUCCESS != err) {
    printf("Cannot create the %s renderer\n", rendererList[0].c_str());
    return NVCV_ERR_FEATURENOTFOUND;
  }

  if (!SetPathIfFileExists("", FLAG_renderModel, path) && !SetPathIfFileExists(FLAG_modelDir, FLAG_renderModel, path)) {
    err = NVCV_ERR_FILE;
    printf("Cannot find %s: %s\n", FLAG_renderModel.c_str(), NvCV_GetErrorStringFromCode(err));
    return err;
  }
  err = _renderer->read(path.c_str());
  if (NVCV_SUCCESS != err) {
    printf("{\"%s\",\"%s\"}: %s\n", FLAG_modelDir.c_str(), FLAG_renderModel.c_str(), NvCV_GetErrorStringFromCode(err));
    return err;
  }
  err = _renderer->init(_renderWidth, _renderHeight, _windowTitle);
  if (NVCV_SUCCESS != err) {
    printf("renderer init: %s\n", NvCV_GetErrorStringFromCode(err));
    return err;
  }

  BAIL_IF_ERR(err = NvCVImage_Alloc(&_srcGpu,    _videoWidth,   _videoHeight, NVCV_BGR,  NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1));
  BAIL_IF_ERR(err = NvCVImage_Alloc(&_srcImg,    _videoWidth,   _videoHeight, NVCV_BGR,  NVCV_U8, NVCV_CHUNKY, NVCV_CPU_PINNED, 0));
  BAIL_IF_ERR(err = NvCVImage_Alloc(&_compImg,   _compWidth,   _compHeight,   NVCV_BGR,  NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 0));
  BAIL_IF_ERR(err = NvCVImage_Alloc(&_renderImg, _renderWidth, _renderHeight, NVCV_RGBA, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 0));
  CVWrapperForNvCVImage(&_compImg, &_ocvDstImg);
  CVWrapperForNvCVImage(&_srcImg,  &_ocvSrcImg);
  resizeDst();

  if (!FLAG_outFile.empty() || !FLAG_show) {
    err = App::openOutputVideo(StringToFourcc(FLAG_codec), _frameRate, _compWidth, _compHeight);
    if (NVCV_SUCCESS != err) {
      printf("ERROR: \"%s\": %s\n", _outFile.c_str(), getErrorStringFromCode(err));
      goto bail;
    }
  }

  // Initialize AR effect
  switch (FLAG_exprMode) {
    default:
      printf("Unknown expression mode %u; using 1=mesh instead\n", FLAG_exprMode);
      /* fall through */
    case EXPR_MODE_MESH:
      BAIL_IF_ERR(err = initFaceFit());
      break;
    case EXPR_MODE_MLP:
      BAIL_IF_ERR(err = initMLPExpressions());
      break;
  }

  if (FLAG_show) cv::namedWindow(_windowTitle, 1);

#if _ENABLE_UI
  if(FLAG_showUI)
    ui_obj_.init(_exprCount, _filtering, FLAG_exprMode, _viewMode, _showFPS);
#endif  // _ENABLE_UI

bail:
  return err;
}


NvCV_Status App::overlayLandmarks(const float landmarks[126 * 2], unsigned screenHeight, NvCVImage* im) {
  cv::Mat frame;
  CVWrapperForNvCVImage(im, &frame);
  float scale = (float)sqrtf((float)frame.rows / screenHeight);
  if (scale < 1.f) scale = 1.f;
  int size = int(5 * scale), thickness = int(scale);
  const float* p = landmarks;
  for (unsigned i = 126; i--; p += 2)
    cv::drawMarker(frame, { (int)lround(p[0]), (int)lround(p[1]) }, CV_RGB(0, 255, 0), cv::MARKER_CROSS, size, thickness, 8);
  return NVCV_SUCCESS;
}

NvCV_Status App::run() {
  NvCV_Status err = NVCV_SUCCESS;
  NvCVImage tmpImg, view;

  for (unsigned frameCount = 0;; ++frameCount) {
    if (!_vidIn.read(_ocvSrcImg) || _ocvSrcImg.empty()) {
      if (!frameCount) return NvFromAppErr(APP_ERR_VIDEO);  // No frames in video
      if (!FLAG_loop)  return NvFromAppErr(APP_ERR_EOF);    // Video has completed
      _vidIn.set(CV_CAP_PROP_POS_FRAMES, 0);                // Rewind, because loop mode has been selected
      --frameCount;                                         // Account for the wasted frame
      continue;                                             // Read the first frame again
    }

#ifdef _ENABLE_UI
    unsigned int exprMode = 0;
    bool uncalibrate = false;
    bool calibrate = false;
    unsigned int filter = 0;
    unsigned int viewMode = 0;
    bool killApp = false;
    if (FLAG_showUI) {
      ui_obj_.stateQuerybyCore(viewMode, exprMode, filter, calibrate, uncalibrate, _showFPS, _globalExpressionParam, _expressionZeroPoint, _expressionScale, _expressionExponent, killApp);

      if (killApp == true) {
        return NvFromAppErr(APP_ERR_CANCEL);
      }
      _performCalibration = calibrate;

      if (viewMode != _viewMode) {
        _viewMode = viewMode;
        resizeDst();
      }
      if (uncalibrate) {
        unCalibrateExpressionWeights();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_FACE_BOX) {
        toggleFaceBoxFiltering();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS) {
        toggleLandmarkFiltering();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE) {
        togglePoseFiltering();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS) {
        toggleExpressionFiltering();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_FACIAL_GAZE) {
        toggleGazeFiltering();
      }
      if ((filter ^ _filtering) & NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS) {
        toggleClosureEnhancement();
      }
      _filtering = filter;

      if (_exprMode != exprMode) {
        if (exprMode == EXPR_MODE_MESH) {
          initFaceFit();
        }
        else if (exprMode == EXPR_MODE_MLP) {
          initMLPExpressions();
        }
        else {
          // add more modes if needed
        }
      }
    }
#endif  // _ENABLE_UI

    BAIL_IF_ERR(err = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _stream, nullptr));
    BAIL_IF_ERR(err = NvAR_Run(_featureHan));
    unsigned isFaceDetected = (_outputBboxes.num_boxes > 0) ? 0xFF : 0;
    if (_cameraNeedsUpdate) {
      err = updateCamera();
      if (NVCV_SUCCESS != err) {
        printf("renderer setCamera: %s\n", NvCV_GetErrorStringFromCode(err));
        return err;
      }
    }
    if (_performCalibration) {
      calibrateExpressionWeights();
    }
    normalizeExpressionsWeights();
    if (_viewMode & VIEW_LM & isFaceDetected) {
      BAIL_IF_ERR(err = overlayLandmarks(&_landmarks.data()->x, _renderHeight, &_srcImg));
    }
    if (_viewMode & VIEW_IMAGE) {
      NvCVImage_InitView(&view, &_compImg, _miniX, _miniY, _miniWidth, _miniHeight);
      err = ResizeNvCVImage(&_srcImg, &view);
    }
    if (_viewMode & VIEW_PLOT) {
      if (!isFaceDetected)  std::fill(_expressions.begin(), _expressions.end(), 0);
      barPlotExprs();
    }
    if (_viewMode & VIEW_MESH) {
      if (isFaceDetected) {
        float* head_translation = _poseMode == 1 ? &_pose.translation.vec[0] : nullptr;
        BAIL_IF_ERR(err = _renderer->render(_expressions.data(), &_pose.rotation.x, head_translation, &_renderImg));   // GL _renderImg is upside down
        NvCVImage_InitView(&view, &_compImg, _renderX, _renderY, _renderWidth, _renderHeight);
        NvCVImage_FlipY(&view, &view);  // Since OpenGL renderImg is upside-down, we copy it to a flipped dst
        NvCVImage_Transfer(&_renderImg, &view, 1.0f, _stream, nullptr);  // VFlip RGBA --> BGR
      }
      else {
        cv::Mat compImgCVMat;
        CVWrapperForNvCVImage(&_compImg, &compImgCVMat);
        cv::rectangle(compImgCVMat, cv::Rect(_renderX, _renderY, _renderWidth, _renderHeight), cv::Scalar(0, 0, 0), -1);
      }
    }
    if (_vidOut.isOpened())
      _vidOut.write(_ocvDstImg);
    drawFPS(_ocvDstImg);
    if (FLAG_show && _ocvDstImg.cols && _ocvDstImg.rows) {
      cv::imshow(_windowTitle, _ocvDstImg);
    }

    int key = cv::waitKey(1);
    if (key >= 0 && FLAG_debug)
      printf("Key press '%c' (%02x)\n", ((0x20 <= key && key <= 0x7f) ? key : '#'), key);
#ifdef _ENABLE_UI
    if (FLAG_showUI){
      ui_obj_.stateSetbyCore(_expressions, _expressionZeroPoint, _expressionScale, _expressionExponent, (uncalibrate || calibrate), key);
    }
#endif  // _ENABLE_UI
    if (!FLAG_showUI) {
      switch (key) {
      case 27 /*ESC*/:
      case 'q': case 'Q':       return NvFromAppErr(APP_ERR_CANCEL);  // Quit
      case 'i':                 _viewMode ^= VIEW_IMAGE;  resizeDst();  break;
      case 'l':                 _viewMode ^= VIEW_LM;     resizeDst();  break;
      case 'm':                 _viewMode ^= VIEW_MESH;   resizeDst();  break;
      case 'n':                 _performCalibration = true;             break;
      case 'p':                 _viewMode ^= VIEW_PLOT;   resizeDst();  break;
      case 'f':                 _showFPS = !_showFPS;                   break;
      case '1':                 initFaceFit();                          break;
      case '2':                 initMLPExpressions();                   break;
      case 'L': case CTL('L'):  toggleLandmarkFiltering();              break;
      case 'N': case CTL('N'):  unCalibrateExpressionWeights();         break;
      case 'P': case CTL('P'):  togglePoseFiltering();                  break;
      case 'E': case CTL('E'):  toggleExpressionFiltering();            break;
      case 'G': case CTL('G'):  toggleGazeFiltering();                  break;
      case 'C': case CTL('C'):  toggleClosureEnhancement();             break;
      case 'M': case CTL('M'):  togglePoseMode();                       break;
      default:
        if (key < 0) continue;                // No key
        break;                                // Non-mapped key
      }
    }
  }
bail:
  return err;
}

NvCV_Status App::resizeDst() {
  NvCV_Status err = NVCV_SUCCESS;
  unsigned width = 0, height = 0;
  if (_viewMode & VIEW_IMAGE) {
    width += _miniWidth;
    if (height < _miniHeight)
      height = _miniHeight;
    _miniX = 0;
    _miniY = 0;
  }
  if (_viewMode & VIEW_MESH) {
    width += _renderWidth;
    if (height < _renderHeight)
      height = _renderHeight;
    _renderY = 0;
    _renderX = (_viewMode & VIEW_IMAGE) ? _miniWidth : 0;
  }
  if (_viewMode & VIEW_PLOT) {
    _plotX = 0;
    _plotY = height;
    _plotWidth = width;
    height += _plotHeight;
  }
  BAIL_IF_ERR(err = NvCVImage_Realloc(&_compImg, width, height, _compImg.pixelFormat, _compImg.componentType,
      NVCV_CHUNKY, NVCV_CPU, 0));
  memset(_compImg.deletePtr, 0, _compImg.bufferBytes);
  CVWrapperForNvCVImage(&_compImg, &_ocvDstImg);
  cv::resizeWindow(_windowTitle, width, height);

bail:
  return err;
}

void App::barPlotExprs() {
  int         barWidth  = (int)(_plotWidth / _expressions.size()),
              barHeight = (int)_plotHeight;
  cv::Scalar  fgColor, bgColor, txColor;
  cv::Rect    r;
  cv::Point   pt;
  std::string str;
  const char  *(*exprAbbr)[4] = (_expressions.size() > 6) ? _exprAbbr : _sfmExprAbbr;

  bgColor = { 0, 0, 0, 255 };
  r     = { _plotX, _plotY, (int)_plotWidth, (int)_plotHeight };
  cv::rectangle(_ocvDstImg, r, bgColor, cv::FILLED, cv::LINE_4, 0);
  bgColor   = { 32, 32, 32, 255 };
  fgColor   = { 0, 255, 0, 255 };
  txColor   = CV_RGB(255, 255, 0); //{ 255, 255, 0, 255 };
  r.width = barWidth - 1;
  for (unsigned i = unsigned(_expressions.size()); i--;) {
    r.x = _plotX + i * barWidth;
    r.y = _plotY;
    r.height = _plotHeight;
    cv::rectangle(_ocvDstImg, r, bgColor, cv::FILLED, cv::LINE_4, 0);
    r.height = (int)(_expressions[i] * barHeight + 0.5f);
    r.y = _plotY + _plotHeight - r.height;
    cv::rectangle(_ocvDstImg, r, fgColor, cv::FILLED, cv::LINE_4, 0);
    cv::putText(_ocvDstImg, std::to_string(i), cv::Point(r.x + 1, _plotY + 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.25, txColor, 1, cv::LINE_8, false);
    for (unsigned j = 0; j < 4; ++j)
      if (exprAbbr[i][j])
        cv::putText(_ocvDstImg, exprAbbr[i][j], cv::Point(r.x + 1, _plotY + 20 + 10 * j),
            cv::FONT_HERSHEY_SIMPLEX, 0.25, txColor, 1, cv::LINE_8, false);
  }
}

void App::getFPS() {
  const float timeConstant = 16.f;
  _timer.stop();
  float t = (float)_timer.elapsedTimeFloat();
  if (t < 100.f) {
    if (_frameTime)
      _frameTime += (t - _frameTime) * (1.f / timeConstant);  // 1 pole IIR filter
    else
      _frameTime = t;
  }
  else {            // Ludicrous time interval; reset
    _frameTime = 0.f;  // WAKE UP
  }
  _timer.start();
}

void App::drawFPS(cv::Mat& img) {
  getFPS();
  if (_frameTime && _showFPS) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.1f", 1. / _frameTime);
    cv::putText(img, buf, cv::Point(img.cols - 80, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
  }
}

NvCV_Status App::toggleFaceBoxFiltering() {
  _filtering ^= NVAR_TEMPORAL_FILTER_FACE_BOX;
  NvCV_Status  err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("FaceBox Filtering %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_FACE_BOX) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::toggleLandmarkFiltering() {
  _filtering ^= NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS;
  NvCV_Status  err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("Landmark Filtering %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::togglePoseFiltering() {
  _filtering ^= NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE;
  NvCV_Status err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("Pose Filtering %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::toggleExpressionFiltering() {
  _filtering ^= NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS;
  NvCV_Status err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("Expression Filtering %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::toggleGazeFiltering() {
  _filtering ^= NVAR_TEMPORAL_FILTER_FACIAL_GAZE;
  NvCV_Status err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("Gaze Filtering %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_FACIAL_GAZE) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::toggleClosureEnhancement() {
  _filtering ^= NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS;
  NvCV_Status err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
  if (NVCV_SUCCESS == err)
    err = NvAR_Load(_featureHan);
  if (FLAG_verbose)
    printf("Closure Enhancement %s\n", ((_filtering & NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS) ? "ON" : "OFF"));
  return err;
}

NvCV_Status App::togglePoseMode() {
  _poseMode = !_poseMode;
  NvCV_Status err = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(PoseMode), _poseMode);
  if (NVCV_SUCCESS == err) err = NvAR_Load(_featureHan);
  _cameraNeedsUpdate = true;
  return err;
}

/********************************************************************************
 * main
 ********************************************************************************/

int main(int argc, char **argv) {
  NvCV_Status err = NVCV_SUCCESS;
  App app;
  int nErrs;

  if (0 != (nErrs = ParseMyArgs(argc, argv))) {
    if (HELP_REQUESTED == nErrs)          // If it was  a call for help ...
      BAIL(err, NVCV_SUCCESS);            // ... just exit quietly
    printf("ERROR: argument syntax\n");
    BAIL(err, NVCV_ERR_PARSE);
  }

  if (FLAG_fitModel.empty())
    FLAG_fitModel = DEFAULT_FACE_MODEL;
  if (FLAG_renderModel.empty())
    FLAG_renderModel = DEFAULT_RENDER_MODEL;
  if (FLAG_modelDir.empty()) {
    do {
      if (SetDirIfFileExists("../data",       FLAG_renderModel, FLAG_modelDir))  break;
      if (SetDirIfFileExists("../../data",    FLAG_renderModel, FLAG_modelDir))  break;
      if (SetDirIfFileExists("../../../data", FLAG_renderModel, FLAG_modelDir))  break;
      if (SetDirIfFileExists("data",          FLAG_renderModel, FLAG_modelDir))  break;
      printf("Please set --model_dir=/path/to/models\n");
      BAIL(err, NVCV_ERR_MISSINGINPUT);
    } while (0);
  }


  if (!FLAG_inFile.empty()) {                 // Input from a video file
    err = app.setInputVideo(FLAG_inFile);
    if (NVCV_SUCCESS != err) {
      printf("ERROR: \"%s\": %s\n", FLAG_inFile.c_str(), app.getErrorStringFromCode(err));
      goto bail;
    }
  } else {                                    // Input from a webcam, #0
    err = app.setInputCamera(0, FLAG_camRes);
    if (NVCV_SUCCESS != err) {
      printf("ERROR: cam0: %s\n", app.getErrorStringFromCode(err));
      goto bail;
    }
  }
  if (!FLAG_outFile.empty()) {                // Output to a file
    err = app.setOutputVideo(FLAG_outFile);
    if (NVCV_SUCCESS != err) {
      printf("ERROR: \"%s\": %s\n", FLAG_outFile.c_str(), app.getErrorStringFromCode(err));
      goto bail;
    }
  }
  else if (!FLAG_show) {  // Without choosing to show or write to an output file, it would be wasting energy
    printf("WARNING: Setting --show since neither --show nor --out were supplied\n");
    FLAG_show = true;
  }
  if (FLAG_verbose) printf("Enabled filters = %x\n", FLAG_filter);
  if (FLAG_verbose) printf("Enabled cnn gaze = %x\n", FLAG_gaze);
  err = app.init();
  BAIL_IF_ERR(err);

  err = app.run();
  switch ((int)err) {
    case App::APP_ERR_CANCEL: // The user stopped
    case App::APP_ERR_EOF:    // The end of the file was reached.
      err = NVCV_SUCCESS;
      break;
    default:
      break;
  }
  BAIL_IF_ERR(err);

bail:
  if (err)
    printf("ERROR: %s\n", app.getErrorStringFromCode(err));
  return (int)err;
}
