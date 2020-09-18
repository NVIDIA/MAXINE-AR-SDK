/*###############################################################################
#
# Copyright(c) 2019 NVIDIA CORPORATION.All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
###############################################################################*/

#include "FeatureVertexName.h"
#include <stdlib.h>
#include <string.h>


// TODO: We should read this from a file
const LandmarkEOSMap LandmarkMapEOS[] = {
    {   33, "chin bottom" },
    {  225, "right eyebrow outer-corner" },
    {  229, "right eyebrow between middle and outer corner" },
    {  233, "right eyebrow middle, vertical middle" },
    { 2086, "right eyebrow between middle and inner corner" },
    {  157, "right eyebrow inner-corner" },
    {  590, "left eyebrow inner-corner" },
    { 2091, "left eyebrow between inner corner and middle" },
    {  666, "left eyebrow middle" },
    {  662, "left eyebrow between middle and outer corner" },
    {  658, "left eyebrow outer-corner" },
    { 2842, "bridge of the nose (parallel to upper eye lids)" },
    {  379, "middle of the nose, a bit below the lower eye lids" },
    {  272, "above nose-tip (1cm or so)" },
    {  114, "nose-tip" },
    {  100, "right nostril, below nose, nose-lip junction" },
    { 2794, "nose-lip junction" },
    {  270, "nose-lip junction" },
    { 2797, "nose-lip junction" },
    {  537, "left nostril, below nose, nose-lip junction" },
    {  177, "right eye outer-corner" },
    {  172, "right eye pupil top right (from subject's perspective)" },
    {  191, "right eye pupil top left" },
    {  181, "right eye inner-corner" },
    {  173, "right eye pupil bottom left" },
    {  174, "right eye pupil bottom right" },
    {  614, "left eye inner-corner" },
    {  624, "left eye pupil top right" },
    {  605, "left eye pupil top left" },
    {  610, "left eye outer-corner" },
    {  607, "left eye pupil bottom left" },
    {  606, "left eye pupil bottom right" },
    {  398, "right mouth corner" },
    {  315, "upper lip right top outer" },
    {  413, "upper lip middle top right" },
    {  329, "upper lip middle top" },
    {  825, "upper lip middle top left" },
    {  736, "upper lip left top outer" },
    {  812, "left mouth corner" },
    {  841, "lower lip left bottom outer" },
    {  693, "lower lip middle bottom left" },
    {  411, "lower lip middle bottom" },
    {  264, "lower lip middle bottom right" },
    {  431, "lower lip right bottom outer" },
    {  416, "upper lip right bottom outer" },
    {  423, "upper lip middle bottom" },
    {  828, "upper lip left bottom outer" },
    {  817, "lower lip left top outer" },
    {  442, "lower lip middle top" },
    {  404, "lower lip right top outer" },
    {  0xFFFF,  nullptr }
};

static const LandmarksMap LandmarkMap[] = {
    {  0,  0, "right contour point 1"  },
    {  1,  2, "right contour point 2"  },
    {  2,  4, "right contour point 3"  },
    {  3,  6, "right contour point 4"  },
    {  4,  8, "right contour point 5"  },
    {  5,  10, "right contour point 6"  },
    {  6,  12, "right contour point 7"  },
    {  7,  14, "right contour point 8"  },
    {  8,  16, "chin bottom"  },
    {  9,  18, "left contour point 1"  },
    {  10,  20, "left contour point 2"  },
    {  11,  22, "left contour point 3"  },
    {  12,  24, "left contour point 4"  },
    {  13,  26, "left contour point 5"  },
    {  14,  28, "left contour point 6"  },
    {  15,  30, "left contour point 7"  },
    {  16,  32, "left contour point 8"  },
    {  17,  33, "right eyebrow outer-corner"  },
    {  18,  34, "right eyebrow between middle and outer corner"  },
    {  19,  35, "right eyebrow middle, vertical middle"  },
    {  20,  36, "right eyebrow between middle and inner corner"  },
    {  21,  37, "right eyebrow inner-corner"  },
    {  22,  42, "left eyebrow inner-corner"  },
    {  23,  43, "left eyebrow between inner corner and middle"  },
    {  24,  44, "left eyebrow middle"  },
    {  25,  45, "left eyebrow between middle and outer corner"  },
    {  26,  46, "left eyebrow outer-corner"  },
    {  27,  51, "bridge of the nose (parallel to upper eye lids)"  },
    {  28,  52, "middle of the nose, a bit below the lower eye lids"  },
    {  29,  53, "above nose-tip (1cm or so)"  },
    {  30,  54, "nose-tip"  },
    {  31,  57, "right nostril, below nose, nose-lip junction"  },
    {  32,  58, "nose-lip junction"  },
    {  33,  59, "nose-lip junction"  },
    {  34,  60, "nose-lip junction"  },
    {  35,  61, "left nostril, below nose, nose-lip junction"  },
    {  36,  64, "right eye outer-corner"  },
    {  37,  65, "right eye pupil top right (from subject's perspective)"  },
    {  38,  67, "right eye pupil top left"  },
    {  39,  68, "right eye inner-corner"  },
    {  40,  69, "right eye pupil bottom left"  },
    {  41,  71, "right eye pupil bottom right"  },
    {  42,  81, "left eye inner-corner"  },
    {  43,  82, "left eye pupil top right"  },
    {  44,  84, "left eye pupil top left"  },
    {  45,  85, "left eye outer-corner"  },
    {  46,  86, "left eye pupil bottom left"  },
    {  47,  88, "left eye pupil bottom right"  },
    {  48,  98, "right mouth corner"  },
    {  49,  99, "upper lip right top outer"  },
    {  50,  100, "upper lip middle top right"  },
    {  51,  101, "upper lip middle top"  },
    {  52,  102, "upper lip middle top left"  },
    {  53,  103, "upper lip left top outer"  },
    {  54,  104, "left mouth corner"  },
    {  55,  105, "lower lip left bottom outer"  },
    {  56,  106, "lower lip middle bottom left"  },
    {  57,  107, "lower lip middle bottom"  },
    {  58,  108, "lower lip middle bottom right"  },
    {  59,  109, "lower lip right bottom outer"  },
    {  60,  110, "right inner mouth corner "},
    {  61,  111, "upper lip right bottom outer"  },
    {  62,  112, "upper lip middle bottom"  },
    {  63,  113, "upper lip left bottom outer"  },
    {  64,  114, "left inner mouth corner"},
    {  65,  115, "lower lip left top outer"  },
    {  66,  116, "lower lip middle top"  },
    {  67,  117, "lower lip right top outer"  },
    { 0xFFFF, 0xFFFF,  nullptr }
};


unsigned short FindEOSLandmarkIndexFromName(const char* name)
{
  if (!name)
    return 0xFFFF;
  switch (name[0]) {
  case '#':                                                       // 1-based index ...
    return (unsigned short)(strtol(name + 1, nullptr, 10) - 1); // ... gets converted into a 0-based index
  case '@':                                                       // 0-based index
    return (unsigned short)strtol(name + 1, nullptr, 10);
  default:
    break;
  }
  const LandmarkEOSMap* lmList = LandmarkMapEOS;
  for (; lmList->name != nullptr; ++lmList)
    if (!strcmp(name, lmList->name))
      break;
  return lmList->index;
}

unsigned short FindLandmarkIndexFromName(const unsigned int numLandmarks, const char* name) {
  if (!name)
    return 0xFFFF;
  switch (name[0]) {
  case '#':                                                       // 1-based index ...
    return (unsigned short)(strtol(name + 1, nullptr, 10) - 1); // ... gets converted into a 0-based index
  case '@':                                                       // 0-based index
    return (unsigned short)strtol(name + 1, nullptr, 10);
  default:
    break;
  }
  const LandmarksMap* lmList = LandmarkMap;
  for (; lmList->name != nullptr; ++lmList)
    if (!strcmp(name, lmList->name))
      break;
  if (numLandmarks == 68) {
    return lmList->index_68;
  } else if (numLandmarks == 126) {
    return lmList->index_126;
  } else {
    return 0xFFFF;
  }
}
