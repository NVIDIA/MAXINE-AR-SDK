/*###############################################################################
#
# Copyright(c) 2020 NVIDIA CORPORATION.All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################*/

#if _ENABLE_UI

#include <map>
#include <fstream>
#include <json.hpp>
#include <imgui_internal.h>
#include "nvAR_defs.h"
#include "ExpressionAppUI.h"

#define CTL(x)                      ((x) & 0x1F)

static std::map<int, std::string> exprMap = {
 {0  ,"browDown_L      "},
 {1  ,"browDown_R      "},
 {2  ,"browInnerUp_L   "},
 {3  ,"browInnerUp_R   "},
 {4  ,"browOuterUp_L   "},
 {5  ,"browOuterUp_R   "},
 {6  ,"cheekPuff_L     "},
 {7  ,"cheekPuff_R     "},
 {8  ,"cheekSquint_L   "},
 {9  ,"cheekSquint_R   "},
 {10 ,"eyeBlink_L      "},
 {11 ,"eyeBlink_R      "},
 {12 ,"eyeLookDown_L   "},
 {13 ,"eyeLookDown_R   "},
 {14 ,"eyeLookIn_L     "},
 {15 ,"eyeLookIn_R     "},
 {16 ,"eyeLookOut_L    "},
 {17 ,"eyeLookOut_R    "},
 {18 ,"eyeLookUp_L     "},
 {19 ,"eyeLookUp_R     "},
 {20 ,"eyeSquint_L     "},
 {21 ,"eyeSquint_R     "},
 {22 ,"eyeWide_L       "},
 {23 ,"eyeWide_R       "},
 {24 ,"jawForward      "},
 {25 ,"jawLeft         "},
 {26 ,"jawOpen         "},
 {27 ,"jawRight        "},
 {28 ,"mouthClose      "},
 {29 ,"mouthDimple_L   "},
 {30 ,"mouthDimple_R   "},
 {31 ,"mouthFrown_L    "},
 {32 ,"mouthFrown_R    "},
 {33 ,"mouthFunnel     "},
 {34 ,"mouthLeft       "},
 {35 ,"mouthLowerDown_L"},
 {36 ,"mouthLowerDown_R"},
 {37 ,"mouthPress_L    "},
 {38 ,"mouthPress_R    "},
 {39 ,"mouthPucker     "},
 {40 ,"mouthRight      "},
 {41 ,"mouthRollLower  "},
 {42 ,"mouthRollUpper  "},
 {43 ,"mouthShrugLower "},
 {44 ,"mouthShrugUpper "},
 {45 ,"mouthSmile_L    "},
 {46 ,"mouthSmile_R    "},
 {47 ,"mouthStretch_L  "},
 {48 ,"mouthStretch_R  "},
 {49 ,"mouthUpperUp_L  "},
 {50 ,"mouthUpperUp_R  "},
 {51 ,"noseSneer_L     "},
 {52 ,"noseSneer_R     "}
};

static const unsigned int browStartIndex  = 0;
static const unsigned int browEndIndex    = 5;
static const unsigned int cheekStartIndex = 6;
static const unsigned int cheekEndIndex   = 9;
static const unsigned int eyeStartIndex   = 10;
static const unsigned int eyeEndIndex     = 23;
static const unsigned int jawStartIndex   = 24;
static const unsigned int jawEndIndex     = 27;
static const unsigned int mouthStartIndex = 28;
static const unsigned int mouthEndIndex   = 50;
static const unsigned int noseStartIndex  = 51;
static const unsigned int noseEndIndex    = 52;

static void glfw_error_callback(int error, const char* description) {
  printf("Glfw Error %d: %s\n", error, description);
}

void ExpressionAppUI::init(int numExpr, int filter, int exprMode, int display, int showFPS) {
  keyboard_input_ = -1;
  filter_face_box_ = (filter & (NVAR_TEMPORAL_FILTER_FACE_BOX)) ? true : false;
  filter_face_landmark_ = (filter & (NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS)) ? true : false;
  filter_face_rot_pose_ = (filter & (NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE)) ? true : false;
  filter_face_expr_ = (filter & (NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS)) ? true : false;
  filter_face_gaze_ = (filter & (NVAR_TEMPORAL_FILTER_FACIAL_GAZE)) ? true : false;
  filter_enhance_expr_ = (filter & (NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS)) ? true : false;
  num_expressions_ = numExpr;
  show_expr_   = false;
  brow_expr_  = false;
  cheek_expr_ = false;
  eye_expr_   = false;
  jaw_expr_   = false;
  mouth_expr_ = false;
  nose_expr_  = false;
  curr_state_.expr_mode = exprMode;
  omniverse_interface_window_ = false;
  load_from_file_ = false;
  curr_state_.calibrate = false;
  curr_state_.uncalibrate = false;
  curr_state_.landmark_display = (DISPLAY_LM & display) ? true: false;
  curr_state_.mesh_display = (DISPLAY_MESH & display) ? true : false;
  curr_state_.image_display = (DISPLAY_IMAGE & display) ? true : false;
  curr_state_.bargraph_display = (DISPLAY_PLOT & display) ? true : false;
  curr_state_.expr.resize(numExpr, 0.0f);
  curr_state_.expr_offset.resize(numExpr, 0.0f);
  curr_state_.expr_scale.resize(numExpr, 1.0f);
  curr_state_.expr_exponent.resize(numExpr, 1.0f);
  curr_state_.global_parameter = 1.0f;
  internal_get_state_counter_ = 0;
  internal_set_state_counter_ = 0;
  curr_state_.input_filter = filter;
  curr_state_.show_fps = showFPS;
  show_filter_window_ = false;
  expr = 0;
  ui_state_ = curr_state_;
  ui_state_.expr.resize(numExpr, 0.0f);
  ui_state_.expr_offset.resize(numExpr, 0.0f);
  ui_state_.expr_scale.resize(numExpr, 1.0f);
  ui_state_.expr_exponent.resize(numExpr, 1.0f);
  file_name_ = "";
  ui_keep_running_ = true;
  ui_thread_ = std::thread([this]() { uiRenderThread();});
}

void ExpressionAppUI::cleanup() {
  ui_keep_running_ = false;
  if (ui_thread_.joinable()) {
    ui_thread_.join();
  }
  ui_expression_list_.clear();
}

void ExpressionAppUI::showMLPSetting() {
  ImGui::PushItemWidth(100);
  ImGui::InputInt("Expression Mode : 1 : Mesh Fitting , 2: MLP", &curr_state_.expr_mode);
  if (curr_state_.expr_mode < 1) {
    curr_state_.expr_mode = 1;
  }
  if (curr_state_.expr_mode > 2) {
    curr_state_.expr_mode = 2;
  }
  ImGui::PopItemWidth();
  ImGui::NewLine();
  ImGui::NewLine();
}

void ExpressionAppUI::showFilterSetting() {
  if (!show_filter_window_) {
    if (ImGui::Button("Set Filters")) {
      show_filter_window_ = true;
    }
  }
  if (show_filter_window_) {
    if (ImGui::Button("Close Filter Settings")) {
      show_filter_window_ = false;
    }
  }
  
  if (show_filter_window_) {
    ImGui::SetNextWindowPos({ ImGui::GetCursorPosX() + 100 ,ImGui::GetCursorPosY() + 100 }, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowContentSize(ImVec2(400, 400.0f));
    ImGui::Begin("Filter");
    curr_state_.input_filter = 0;
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_FACE_BOX", &filter_face_box_); ImGui::SameLine();
    ImGui::Text("FILTER_FACE_BOX");
    ImGui::NewLine();
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS", &filter_face_landmark_); ImGui::SameLine();
    ImGui::Text("FILTER_FACIAL_LANDMARKS");
    ImGui::NewLine();
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE", &filter_face_rot_pose_); ImGui::SameLine();
    ImGui::Text("FILTER_FACE_ROTATIONAL_POSE");
    ImGui::NewLine();
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS", &filter_face_expr_); ImGui::SameLine();
    ImGui::Text("FILTER_FACIAL_EXPRESSIONS");
    ImGui::NewLine();
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_FACIAL_GAZE", &filter_face_gaze_); ImGui::SameLine();
    ImGui::Text("FILTER_FACIAL_GAZE");
    ImGui::NewLine();
    ImGui::Checkbox("##NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS", &filter_enhance_expr_); ImGui::SameLine();
    ImGui::Text("FILTER_ENHANCE_EXPRESSIONS");
    ImGui::NewLine();
    ImGui::NewLine();
    
    curr_state_.input_filter |= filter_face_box_ ? NVAR_TEMPORAL_FILTER_FACE_BOX : 0;
    curr_state_.input_filter |= filter_face_landmark_ ? NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS : 0;
    curr_state_.input_filter |= filter_face_rot_pose_ ? NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE : 0;
    curr_state_.input_filter |= filter_face_expr_ ? NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS : 0;
    curr_state_.input_filter |= filter_face_gaze_ ? NVAR_TEMPORAL_FILTER_FACIAL_GAZE : 0;
    curr_state_.input_filter |= filter_enhance_expr_ ? NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS : 0;

    if (ImGui::Button("Close")) {
      show_filter_window_ = false;
    }

    ImGui::End();
  }

  ImGui::NewLine();
  ImGui::NewLine();
}

void ExpressionAppUI::showExpressionWindow() {
  // Fill the list with curently active expressions

  if ((1 << BROW) & expr) {
    for (int i = 0; i <= 5; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(browStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (browEndIndex - browStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }
  if ((1 << CHEEK) & expr) {
    for (int i = 6; i <= 9; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(cheekStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (cheekEndIndex - cheekStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }
  if ((1 << EYE) & expr) {
    for (int i = 10; i <= 23; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(eyeStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (eyeEndIndex - eyeStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }
  if ((1 << JAW) & expr) {
    for (int i = 24; i <= 27; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(jawStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (jawEndIndex - jawStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }
  if ((1 << MOUTH) & expr) {
    for (int i = 28; i <= 50; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(mouthStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (mouthEndIndex - mouthStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }
  if ((1 << NOSE) & expr) {
    for (int i = 51; i <= 52; i++) {
      ui_expression_list_.insert({ i });
    }
  }
  else {
    auto it = ui_expression_list_.find(noseStartIndex);
    if (it != ui_expression_list_.end()) {
      auto begin = it;
      std::advance(it, (noseEndIndex - noseStartIndex + 1));
      ui_expression_list_.erase(begin, it);
    }
  }

  ImGui::SetNextWindowContentSize(ImVec2(1080, 400.0f));
  ImGui::Begin("Expressions");
  ImGui::NewLine();
  ImGui::NewLine();
  ImGui::SliderFloat("Global Expression Parameter: Filter the effect of scaling and offset", &curr_state_.global_parameter, 0.0f, 1.0f);
  ImGui::NewLine();
  ImGui::NewLine();

  expr = 0;

  for (auto it = ui_expression_list_.begin(); it != ui_expression_list_.end(); it++) {
    ImGui::PushItemWidth(400);
    char overlay_buf[32];
    ImFormatString(overlay_buf, IM_ARRAYSIZE(overlay_buf), "%.04f%%", curr_state_.expr[*it]);
    ImGui::ProgressBar(curr_state_.expr[*it], ImVec2(0.0f, 0.0f), overlay_buf);
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    int id = *it;
    ImGui::Text("%s", exprMap[id].c_str());
    ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 50);
    ImGui::PushItemWidth(100);
    ImGui::Text("Scale(0.0-2.0)");
    ImGui::SameLine();
    std::string scaleLabel = "##Scale:" + exprMap[*it];
    ImGui::SliderFloat(scaleLabel.c_str(), &curr_state_.expr_scale[*it], 0.0f, 2.0f);
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10);
    ImGui::Text("Offset(-1.0 to 1.0)");
    ImGui::SameLine();
    std::string offsetLabel = "##Offset:" + exprMap[*it];
    ImGui::SliderFloat(offsetLabel.c_str(), &curr_state_.expr_offset[*it], -1.0f, 1.0f);
    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10);
    ImGui::Text("Exponent(0.0 to 2.0)");
    ImGui::SameLine();
    std::string expLabel = "##Exponent:" + exprMap[*it];
    ImGui::SliderFloat(expLabel.c_str(), &curr_state_.expr_exponent[*it], 0.0f, 2.0f);
    ImGui::PopItemWidth();
  }
  ImGui::End();
}

void ExpressionAppUI::showExpressionPane() {
  ImGui::Text("Expression Graph Options");
  ImGui::Checkbox("Brow", &brow_expr_); ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox("Cheek", &cheek_expr_); ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox("Eye", &eye_expr_); ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox("Jaw", &jaw_expr_); ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox("Mouth", &mouth_expr_); ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::Checkbox("Nose", &nose_expr_);

  expr |= (brow_expr_)  ? (1 << BROW)  : 0;
  expr |= (cheek_expr_) ? (1 << CHEEK) : 0;
  expr |= (eye_expr_)   ? (1 << EYE)   : 0;
  expr |= (jaw_expr_)   ? (1 << JAW)   : 0;
  expr |= (mouth_expr_) ? (1 << MOUTH) : 0;
  expr |= (nose_expr_)  ? (1 << NOSE)  : 0;

  if (expr) {
    show_expr_ = true;
  }
  else {
    if (ui_expression_list_.empty() == false) {
      ui_expression_list_.clear();
    }
    show_expr_ = false;
  }

  if (show_expr_) {
    showExpressionWindow();
  }

  ImGui::NewLine();
  ImGui::NewLine();
}

void ExpressionAppUI::showCalibrationSetting() {
  if (ImGui::Button("Calibrate")) {
    curr_state_.calibrate = true;
  }

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 50);
  if (ImGui::Button("Uncalibrate")) {
    curr_state_.uncalibrate = true;
  }
  ImGui::NewLine();
}


void ExpressionAppUI::showLandmarkOption() {
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 50);
  ImGui::Checkbox("Landmark", &curr_state_.landmark_display);
  ImGui::SameLine();
  ImGui::Checkbox("Mesh", &curr_state_.mesh_display);
  ImGui::SameLine();
  ImGui::Checkbox("Graph", &curr_state_.bargraph_display);
  ImGui::SameLine();
  ImGui::Checkbox("Image", &curr_state_.image_display);
}

void ExpressionAppUI::showImageDisplaySettings() {
  ImGui::Text("Image Settings");
  showLandmarkOption();
  ImGui::NewLine();
}

void ExpressionAppUI::showFPSSetting() {
  ImGui::Checkbox("Toggle FPS Display", &curr_state_.show_fps);
  ImGui::NewLine();
}

void ExpressionAppUI::saveConfigToFile() {
  std::ofstream configFile;

  std::string fileName = "ExpressionAppSettings.json";
  configFile.open(fileName.c_str());

  nlohmann::json settings; {
    settings["MLP"] = curr_state_.expr_mode;
    settings["Filter"] = curr_state_.input_filter;
    settings["Expressions"] = curr_state_.expr;
    settings["ExpressionsOffset"] = curr_state_.expr_offset;
    settings["ExpressionsScale"] = curr_state_.expr_scale;
    settings["ExpressionsExponent"] = curr_state_.expr_exponent;
    settings["GlobalParameter"] = curr_state_.global_parameter;
  }

  configFile << settings;

  configFile.close();
}

void ExpressionAppUI::loadConfgFromFile(const char* filePath) {
  std::ifstream configFile;
  std::string fileName;
  if (!filePath) {
    fileName = "ExpressionAppSettings.json";
  }
  else {
    fileName = filePath;
  }
  configFile.open(fileName.c_str());

  if (configFile) {
    auto settings = nlohmann::json::parse(configFile); {
      curr_state_.expr_mode = settings["MLP"];
      curr_state_.input_filter = settings["Filter"];
      curr_state_.expr = settings["Expressions"].get<std::vector<float>>();
      curr_state_.expr_offset = settings["ExpressionsOffset"].get<std::vector<float>>();
      curr_state_.expr_scale = settings["ExpressionsScale"].get<std::vector<float>>();
      curr_state_.expr_exponent = settings["ExpressionsExponent"].get<std::vector<float>>();
      curr_state_.global_parameter = settings["GlobalParameter"];
    }
  }
  filter_face_box_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_FACE_BOX) ? true : false;
  filter_face_landmark_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS) ? true : false;
  filter_face_rot_pose_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE) ? true : false;
  filter_face_expr_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS) ? true : false;
  filter_face_gaze_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_FACIAL_GAZE) ? true : false;
  filter_enhance_expr_ = (curr_state_.input_filter & NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS) ? true : false;
}

void ExpressionAppUI::openFileLoadSettings() {
  if (load_from_file_) {
    ImGui::SetNextWindowContentSize(ImVec2(500, 100.0f));
    ImGui::Begin("Config Settings");
    ImGui::Text("Enter Full File name with path eg : C:\\sample.json (no double quotes)");
    ImGui::InputText("##FileName", &file_name_);
    ImGui::NewLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 30);
    if (ImGui::Button("OK")) {
      loadConfgFromFile(file_name_.c_str());
      load_from_file_ = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      load_from_file_ = false;
    }
    ImGui::End();
  }
}

void ExpressionAppUI::showSaveSettingsOption() {
  ImGui::NewLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 10);
  if (ImGui::Button("SaveSettings")) {
    saveConfigToFile();
  }
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 50);

  if (ImGui::Button("LoadSettings")) {
    loadConfgFromFile(NULL);
  }
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 50);

  if (load_from_file_) {
    openFileLoadSettings();
    if (ImGui::Button("Close Settings Window")) {
      load_from_file_ = false;
    }
  }
  else {
    if (ImGui::Button("LoadSettingsFromFile")) {
      load_from_file_ = true;
    }
  }
}

void ExpressionAppUI::closeAppSettings() {
  ImGui::NewLine();
  ImGui::NewLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 170);

  if (ImGui::Button("App shutdown")) {
    curr_state_.kill_app_ = true;
  }
}

void ExpressionAppUI::CreateUIElements() {
  showMLPSetting();
  showFilterSetting();
  showExpressionPane();
  showCalibrationSetting();
  showImageDisplaySettings();
  showFPSSetting();
  showSaveSettingsOption();
  closeAppSettings();
}

void ExpressionAppUI::stateQuerybyCore(unsigned int& displayMode, unsigned int& exprMode, unsigned int& filter, bool& calibrate, bool& uncalibrate, bool& showFPS,
  float& globalParam, std::vector<float>& expressionOffset, std::vector<float>& expressionScale, std::vector<float>& expressionExponent, bool& killApp) {
  {
    std::lock_guard<std::mutex>  lock(ui_mutex_);
    displayMode = ((ui_state_.landmark_display == true) ? DISPLAY_LM : 0) + ((ui_state_.mesh_display == true) ? DISPLAY_MESH : 0) + ((ui_state_.bargraph_display == true) ? DISPLAY_PLOT : 0) + ((ui_state_.image_display == true) ? DISPLAY_IMAGE : 0);;
    filter = ui_state_.input_filter;
    calibrate = ui_state_.calibrate;
    uncalibrate = ui_state_.uncalibrate;
    showFPS = ui_state_.show_fps;
    exprMode = ui_state_.expr_mode;
    globalParam = ui_state_.global_parameter;
    expressionOffset = ui_state_.expr_offset;
    expressionScale = ui_state_.expr_scale;
    expressionExponent = ui_state_.expr_exponent;
    killApp = ui_state_.kill_app_;
  }
}

void ExpressionAppUI::stateSetbyCore(std::vector<float> expression,
  std::vector<float> expressionOffset, std::vector<float> expressionScale, std::vector<float> expressionExponent, bool isCalibrated, int key) {
  {
    std::lock_guard<std::mutex>  lock(ui_mutex_);
    ui_state_.expr = expression;
    ui_state_.expr_offset = expressionOffset;
    ui_state_.expr_scale = expressionScale;
    ui_state_.expr_exponent = expressionExponent;
    if ((internal_get_state_counter_ == internal_set_state_counter_) && (isCalibrated)) {
      ui_state_.calibrate = false;
      ui_state_.uncalibrate = false;
    }
    if (key >= 0) {
      keyboard_input_ = key;
    }
  }
}

void ExpressionAppUI::checkForKeyInput() {
  if (keyboard_input_ >= 0) {
    switch (keyboard_input_) {
      case 27 /*ESC*/:
      case 'q': case 'Q':       curr_state_.kill_app_ = true;                               break; // Quit
      case 'i':                 curr_state_.image_display = !ui_state_.image_display;       break;
      case 'l':                 curr_state_.landmark_display = !ui_state_.landmark_display; break;
      case 'm':                 curr_state_.mesh_display = !ui_state_.mesh_display;         break;
      case 'n':                 curr_state_.calibrate = true;                               break;
      case 'p':                 curr_state_.bargraph_display = !ui_state_.bargraph_display; break;
      case 'f':                 curr_state_.show_fps = !ui_state_.show_fps;                 break;
      case '1':                 curr_state_.expr_mode = 1;                                  break;
      case '2':                 curr_state_.expr_mode = 2;                                  break;
      case 'L': case CTL('L'):  filter_face_landmark_ = !filter_face_landmark_;
        curr_state_.input_filter ^= NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS;                  break;
      case 'N': case CTL('N'):  curr_state_.uncalibrate = true;;                            break;
      case 'P': case CTL('P'):  filter_face_rot_pose_ = !filter_face_rot_pose_;
        curr_state_.input_filter ^= NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE;              break;
      case 'E': case CTL('E'):  filter_face_expr_ = !filter_face_expr_;
        curr_state_.input_filter ^= NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS;                break;
      case 'G': case CTL('G'):  filter_face_gaze_ = !filter_face_gaze_;
        curr_state_.input_filter ^= NVAR_TEMPORAL_FILTER_FACIAL_GAZE;                       break;
      case 'C': case CTL('C'):  filter_enhance_expr_ = !filter_enhance_expr_;
        curr_state_.input_filter ^= NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS;               break;
      default:               // No key
        break;
    }
    keyboard_input_ = -1;
  }
}

void ExpressionAppUI::setStateToLocal() {
  std::lock_guard<std::mutex>  lock(ui_mutex_);
  ui_state_ = curr_state_;
  ui_state_.expr = curr_state_.expr;
  ui_state_.expr_exponent = curr_state_.expr_exponent;
  ui_state_.expr_offset = curr_state_.expr_offset;
  ui_state_.expr_scale = curr_state_.expr_scale;
  ui_state_.kill_app_ = curr_state_.kill_app_;
  internal_set_state_counter_ = internal_get_state_counter_;
  if (internal_get_state_counter_ > 100000) {
    internal_get_state_counter_ = internal_set_state_counter_ = 0;
  }
}

void ExpressionAppUI::getStateFromLocal() {
  std::lock_guard<std::mutex>  lock(ui_mutex_);
  curr_state_ = ui_state_;
  curr_state_.expr = ui_state_.expr;
  curr_state_.expr_exponent = ui_state_.expr_exponent;
  curr_state_.expr_offset = ui_state_.expr_offset;
  curr_state_.expr_scale = ui_state_.expr_scale;
  internal_get_state_counter_++;
}

void ExpressionAppUI::uiRenderThread() {
  ImVec4 clear_color_;

  glfwSetErrorCallback(glfw_error_callback);
  glfwInit();
  GLFWwindow* window = glfwCreateWindow(640, 540, "Expression App Interface", NULL, NULL);
  if (window == NULL) {
    GLenum err = glGetError();
    printf("Create window failed : %d", err);
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
    style.WindowRounding = 0.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;
  }
  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  setStateToLocal();
  {
    std::lock_guard<std::mutex>  lock(ui_mutex_);
    internal_get_state_counter_ = 0;
    internal_set_state_counter_ = 0;
  }

  while (!glfwWindowShouldClose(window) && ui_keep_running_) {
    if (!ui_keep_running_) {
      break;
    }
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    {
      ImGui::SetNextWindowContentSize(ImVec2(500, 500.0f));
      ImGui::Begin("Expression: Input Options");
      getStateFromLocal();

      CreateUIElements();
      checkForKeyInput();
      setStateToLocal();
      ImGui::End();
    }
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color_.x * clear_color_.w, clear_color_.y * clear_color_.w, clear_color_.z * clear_color_.w, clear_color_.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      GLFWwindow* backup_current_context = glfwGetCurrentContext();
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);

  // TODO: call terminate if GL renderer is not being used
  // glfwTerminate();
}

#endif

