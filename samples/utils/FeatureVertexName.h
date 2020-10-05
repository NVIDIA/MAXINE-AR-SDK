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

#ifndef __FEATURE_VERTEX_NAME__
#define __FEATURE_VERTEX_NAME__


struct LandmarkEOSMap  { unsigned short index; const char *name; };
struct LandmarksMap { unsigned short index_68; unsigned short index_126; const char* name; };

unsigned short FindEOSLandmarkIndexFromName(const char *name);
unsigned short FindLandmarkIndexFromName(const unsigned int numLandmarks, const char* name);

#endif /* __FEATURE_VERTEX_NAME__ */
