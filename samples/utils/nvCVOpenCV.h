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

#ifndef __NVCVOPENCV_H__
#define __NVCVOPENCV_H__

#include "nvCVImage.h"
#include "opencv2/opencv.hpp"

// Set an OpenCV Mat image from parameters
inline void CVImageSet(cv::Mat *cvIm, int width, int height, int numComps, int compType, int compBytes, void* pixels, size_t rowBytes) {
  size_t pixBytes   = numComps * compBytes;
  size_t widthBytes = width    * pixBytes;
  cvIm->flags = cv::Mat::MAGIC_VAL + (CV_MAKETYPE(compType, numComps) & cv::Mat::TYPE_MASK);
  if (rowBytes == widthBytes)
    cvIm->flags |= cv::Mat::CONTINUOUS_FLAG;
  cvIm->step.p    = cvIm->step.buf;
  cvIm->step[0]   = rowBytes;
  cvIm->step[1]   = pixBytes;
  cvIm->dims      = 2;
  cvIm->size      = cv::MatSize(&cvIm->rows);
  cvIm->rows      = height;
  cvIm->cols      = width;
  cvIm->data      = (uchar*)pixels;
  cvIm->datastart = (uchar*)pixels;
  cvIm->datalimit = cvIm->datastart + rowBytes * height;
  cvIm->dataend   = cvIm->datalimit - rowBytes + widthBytes;
  cvIm->allocator = 0;
  cvIm->u         = 0;
}

// Wrap an NvCVImage in a cv::Mat
inline void CVWrapperForNvCVImage(const NvCVImage *nvcvIm, cv::Mat *cvIm) {
  static const char cvType[] = { 7, 0, 2, 3, 7, 7, 4, 5, 7, 7, 6 };
  CVImageSet(cvIm, nvcvIm->width, nvcvIm->height, nvcvIm->numComponents, cvType[(int)nvcvIm->componentType], nvcvIm->componentBytes, nvcvIm->pixels, nvcvIm->pitch);
}

// Wrap a cv::Mat in an NvCVImage.
inline void NVWrapperForCVMat(const cv::Mat *cvIm, NvCVImage *nvcvIm) {
  static const NvCVImage_PixelFormat nvFormat[] = { NVCV_FORMAT_UNKNOWN, NVCV_Y, NVCV_YA, NVCV_BGR, NVCV_BGRA };
  static const NvCVImage_ComponentType nvType[] = { NVCV_U8, NVCV_TYPE_UNKNOWN, NVCV_U16, NVCV_S16, NVCV_S32, NVCV_F32,
                                                    NVCV_F64, NVCV_TYPE_UNKNOWN };
  nvcvIm->pixels         = cvIm->data;
  nvcvIm->width          = cvIm->cols;
  nvcvIm->height         = cvIm->rows;
  nvcvIm->pitch          = (int)cvIm->step[0];
  nvcvIm->pixelFormat    = nvFormat[cvIm->channels() <= 4 ? cvIm->channels() : 0];
  nvcvIm->componentType  = nvType[cvIm->depth() & 7];
  nvcvIm->bufferBytes    = 0;
  nvcvIm->deletePtr      = nullptr;
  nvcvIm->deleteProc     = nullptr;
  nvcvIm->pixelBytes     = (unsigned char)cvIm->step[1];
  nvcvIm->componentBytes = (unsigned char)cvIm->elemSize1();
  nvcvIm->numComponents  = (unsigned char)cvIm->channels();
  nvcvIm->planar         = NVCV_CHUNKY;
  nvcvIm->gpuMem         = NVCV_CPU;
  nvcvIm->reserved[0]    = 0;
  nvcvIm->reserved[1]    = 0;
}

#endif // __NVCVOPENCV_H__