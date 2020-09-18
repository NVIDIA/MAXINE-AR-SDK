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

#ifndef __NVCVIMAGE_H__
#define __NVCVIMAGE_H__

#include "nvCVStatus.h"

#ifdef __cplusplus
extern "C" {
#endif // ___cplusplus

struct CUstream_st;   // typedef struct CUstream_st *CUstream;

//! The format of pixels in an image.
typedef enum NvCVImage_PixelFormat {
  NVCV_FORMAT_UNKNOWN  = 0,    //!< Unknown pixel format.
  NVCV_Y               = 1,    //!< Luminance (gray).
  NVCV_A               = 2,    //!< Alpha (opacity)
  NVCV_YA              = 3,    //!< { Luminance, Alpha }
  NVCV_RGB             = 4,    //!< { Red, Green, Blue }
  NVCV_BGR             = 5,    //!< { Red, Green, Blue }
  NVCV_RGBA            = 6,    //!< { Red, Green, Blue, Alpha }
  NVCV_BGRA            = 7,    //!< { Red, Green, Blue, Alpha }
  NVCV_YUV420          = 8,    //!< Luminance and subsampled Chrominance { Y, Cb, Cr }
  NVCV_YUV422          = 9,    //!< Luminance and subsampled Chrominance { Y, Cb, Cr }
} NvCVImage_PixelFormat;


//! The data type used to represent each component of an image.
typedef enum NvCVImage_ComponentType {
  NVCV_TYPE_UNKNOWN  = 0,      //!< Unknown type of component.
  NVCV_U8            = 1,      //!< Unsigned 8-bit integer.
  NVCV_U16           = 2,      //!< Unsigned 16-bit integer.
  NVCV_S16           = 3,      //!< Signed 16-bit integer.
  NVCV_F16           = 4,      //!< 16-bit floating-point.
  NVCV_U32           = 5,      //!< Unsigned 32-bit integer.
  NVCV_S32           = 6,      //!< Signed 32-bit integer.
  NVCV_F32           = 7,      //!< 32-bit floating-point (float).
  NVCV_U64           = 8,      //!< Unsigned 64-bit integer.
  NVCV_S64           = 9,      //!< Signed 64-bit integer.
  NVCV_F64           = 10,     //!< 64-bit floating-point (double).
} NvCVImage_ComponentType;


//! Value for the planar field or layout argument. Two values are currently accommodated for RGB:
//! Interleaved or chunky storage locates all components of a pixel adjacent in memory,
//! e.g. RGBRGBRGB... (denoted [RGB]).
//! Planar storage locates the same component of all pixels adjacent in memory,
//! e.g. RRRRR...GGGGG...BBBBB... (denoted [R][G][B])
//! YUV has many more variants.
//! 4:2:2 can be chunky, planar or semi-planar, with different orderings.
//! 4:2:0 can be planar or semi-planar, with different orderings.
//! Aliases are provided for FOURCCs defined at fourcc.org.
//! Note: the LSB can be used to distinguish between chunky and planar formats.
#define NVCV_INTERLEAVED   0   //!< All components of pixel(x,y) are adjacent (same as chunky) (default for non-YUV).
#define NVCV_CHUNKY        0   //!< All components of pixel(x,y) are adjacent (same as interleaved).
#define NVCV_PLANAR        1   //!< The same component of all pixels are adjacent.
#define NVCV_UYVY          2   //!< [UYVY]    Chunky 4:2:2 (default for 4:2:2)
#define NVCV_VYUY          4   //!< [VYUY]    Chunky 4:2:2
#define NVCV_YUYV          6   //!< [YUYV]    Chunky 4:2:2
#define NVCV_YVYU          8   //!< [YVYU]    Chunky 4:2:2
#define NVCV_YUV           3   //!< [Y][U][V] Planar 4:2:2 or 4:2:0
#define NVCV_YVU           5   //!< [Y][V][U] Planar 4:2:2 or 4:2:0
#define NVCV_YCUV          7   //!< [Y][UV]   Semi-planar 4:2:2 or 4:2:0 (default for 4:2:0)
#define NVCV_YCVU          9   //!< [Y][VU]   Semi-planar 4:2:2 or 4:2:0
#define NVCV_YUY2  NVCV_YUYV   //!< [YUYV]    Chunky 4:2:2
#define NVCV_I420  NVCV_YUV    //!< [Y][U][V] Planar 4:2:2 or 4:2:0
#define NVCV_IYUV  NVCV_YUV    //!< [Y][U][V] Planar 4:2:2 or 4:2:0
#define NVCV_YV12  NVCV_YVU    //!< [Y][V][U] Planar 4:2:2 or 4:2:0
#define NVCV_NV12  NVCV_YCUV   //!< [Y][UV]   Semi-planar 4:2:2 or 4:2:0 (default for 4:2:0)
#define NVCV_NV21  NVCV_YCVU   //!< [Y][VU]   Semi-planar 4:2:2 or 4:2:0

//! The following are ORed together for the colorspace field for YUV.
//! NVCV_601 and NVCV_709 describe the color axes of YUV.
//! NVCV_VIDEO_RANGE and NVCV_VIDEO_RANGE describe the range, [16, 235] or [0, 255], respectively.
//! NVCV_CHROMA_COSITED and NVCV_CHROMA_INTSTITIAL describe the location of the chroma samples.
#define NVCV_601               0   //!< The Rec.601 YUV colorspace, typically used for SD.
#define NVCV_709               1   //!< The Rec.709 YUV colorspace, typically used for HD.
#define NVCV_VIDEO_RANGE       0   //!< The video range is [16, 235].
#define NVCV_FULL_RANGE        4   //!< The video range is [ 0, 255].
#define NVCV_CHROMA_COSITED    0   //!< The chroma is sampled at the same location as the luma samples horizontally.
#define NVCV_CHROMA_INTSTITIAL 8   //!< The chroma is sampled between luma samples horizontally.
#define NVCV_CHROMA_MPEG2      NVCV_CHROMA_COSITED
#define NVCV_CHROMA_MPEG1      NVCV_CHROMA_INTSTITIAL

//! This is the value for the gpuMem field or the memSpace argument.
#define NVCV_CPU         0   //!< The buffer is stored in CPU memory.
#define NVCV_GPU         1   //!< The buffer is stored in CUDA memory.
#define NVCV_CUDA        1   //!< The buffer is stored in CUDA memory.
#define NVCV_CPU_PINNED   2   //!< The buffer is stored in pinned CPU memory.

//! Image descriptor.
typedef struct
#ifdef _MSC_VER
__declspec(dllexport)
#endif // _MSC_VER
NvCVImage {
  unsigned int              width;                  //!< The number of pixels horizontally in the image.
  unsigned int              height;                 //!< The number of pixels  vertically  in the image.
  signed int                pitch;                  //!< The byte stride between pixels vertically.
  NvCVImage_PixelFormat     pixelFormat;            //!< The format of the pixels in the image.
  NvCVImage_ComponentType   componentType;          //!< The data type used to represent each component of the image.
  unsigned char             pixelBytes;             //!< The number of bytes in a chunky pixel.
  unsigned char             componentBytes;         //!< The number of bytes in each pixel component.
  unsigned char             numComponents;          //!< The number of components in each pixel.
  unsigned char             planar;                 //!< NVCV_CHUNKY, NVCV_PLANAR, NVCV_UYVY, ....
  unsigned char             gpuMem;                 //!< NVCV_CPU, NVCV_CPU_PINNED, NVCV_CUDA, NVCV_GPU
  unsigned char             colorspace;             //!< an OR of colorspace, range and chroma phase.
  unsigned char             reserved[2];            //!< For structure padding and future expansion. Set to 0.
  void                      *pixels;                //!< Pointer to pixel(0,0) in the image.
  void                      *deletePtr;             //!< Buffer memory to be deleted (can be NULL).
  void                      (*deleteProc)(void *p); //!< Delete procedure to call rather than free().
  unsigned long long        bufferBytes;            //!< The maximum amount of memory available through pixels.


#ifdef __cplusplus

  //! Default constructor: fill with 0.
  inline NvCVImage();

  //! Allocation constructor.
  //! \param[in]  width     the number of pixels horizontally.
  //! \param[in]  height    the number of pixels vertically.
  //! \param[in]  format    the format of the pixels.
  //! \param[in]  type      the type of each pixel component.
  //! \param[in]  layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
  //! \param[in]  memSpace  One of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
  //! \param[in]  alignment row byte alignment. Choose 0 or a power of 2.
  //!                       1: yields no gap whatsoever between scanlines;
  //!                       0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
  //!                       Other common values are 16 or 32 for cache line size.
  inline NvCVImage(unsigned width, unsigned height, NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
          unsigned layout = NVCV_CHUNKY, unsigned memSpace = NVCV_CPU, unsigned alignment = 0);

  //! Subimage constructor.
  //! \param[in]  fullImg   the full image, from which this subImage view is to be created.
  //! \param[in]  x         the left edge of the subImage, in reference to the full image.
  //! \param[in]  y         the top edge  of the subImage, in reference to the full image.
  //! \param[in]  width     the width  of the subImage, in pixels.
  //! \param[in]  height    the height of the subImage, in pixels.
  //! \bug        This does not work for planar or semi-planar formats, neither RGB nor YUV.
  //! \note       This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
  inline NvCVImage(NvCVImage *fullImg, int x, int y, unsigned width, unsigned height);

  //! Destructor
  inline ~NvCVImage();

  //! Copy a rectangular subimage. This works for CPU->CPU, CPU->GPU, GPU->GPU, and GPU->CPU.
  //! \param[in]  src     The source image from which to copy.
  //! \param[in]  srcX    The left coordinate of the src rectangle.
  //! \param[in]  srcY    The top  coordinate of the src rectangle.
  //! \param[in]  dstX    The left coordinate of the dst rectangle.
  //! \param[in]  dstY    The top  coordinate of the dst rectangle.
  //! \param[in]  width   The width  of the rectangle to be copied, in pixels.
  //! \param[in]  height  The height of the rectangle to be copied, in pixels.
  //! \note   NvCVImage_Transfer() can handle more cases.
  //! \return NVCV_SUCCESS         if successful
  //! \return NVCV_ERR_MISMATCH    if the formats are different
  //! \return NVCV_ERR_CUDA        if a CUDA error occurred
  //! \return NVCV_ERR_PIXELFORMAT if the pixel format is not yet accommodated.
  //! \bug        This does not work for planar or semi-planar formats, neither RGB nor YUV.
  //! \note       This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
  inline NvCV_Status copyFrom(const NvCVImage *src, int srcX, int srcY, int dstX, int dstY, unsigned width, unsigned height);

  //! Copy from one image to another. This works for CPU->CPU, CPU->GPU, GPU->GPU, and GPU->CPU.
  //! \param[in]  src     The source image from which to copy.
  //! \note   NvCVImage_Transfer() can handle more cases.
  //! \return NVCV_SUCCESS         if successful
  //! \return NVCV_ERR_MISMATCH    if the formats are different
  //! \return NVCV_ERR_CUDA        if a CUDA error occurred
  //! \return NVCV_ERR_PIXELFORMAT if the pixel format is not yet accommodated.
  inline NvCV_Status copyFrom(const NvCVImage *src);

#endif // ___cplusplus
} NvCVImage;


//! Initialize an image. The C++ constructors can initialize this appropriately.
//! This is called by the C++ constructor, but C code should call this explicitly.
//! \param[in,out]  im        the image to initialize.
//! \param[in]      width     the desired width  of the image, in pixels.
//! \param[in]      height    the desired height of the image, in pixels.
//! \param[in]      pitch     the byte stride between pixels vertically.
//! \param[in]      pixels    a pointer to the pixel buffer.
//! \param[in]      format    the format of the pixels.
//! \param[in]      type      the type of the components of the pixels.
//! \param[in]      layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
//! \param[in]      memSpace  Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
//! \return NVCV_SUCCESS         if successful
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not yet accommodated.
NvCV_Status NvCV_API NvCVImage_Init(NvCVImage *im, unsigned width, unsigned height, int pitch, void *pixels,
  NvCVImage_PixelFormat format, NvCVImage_ComponentType type, unsigned layout, unsigned memSpace);


//! Initialize a view into a subset of an existing image.
//! No memory is allocated -- the fullImg buffer is used.
//! \param[in]  subImg  the sub-image view into the existing full image.
//! \param[in]  fullImg the existing full image.
//! \param[in]  x       the left edge of the sub-image, as coordinate of the full image.
//! \param[in]  y       the top  edge of the sub-image, as coordinate of the full image.
//! \param[in]  width   the desired width  of the subImage, in pixels.
//! \param[in]  height  the desired height of the subImage, in pixels.
//! \bug        This does not work for planar or semi-planar formats, neither RGB nor YUV.
//! \note       This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
void NvCV_API NvCVImage_InitView(NvCVImage *subImg, NvCVImage *fullImg, int x, int y, unsigned width, unsigned height);


//! Allocate memory for, and initialize an image. This assumes that the image data structure has nothing meaningful in it.
//! This is called by the C++ constructor, but C code can call this to allocate an image.
//! \param[in,out]  im        the image to initialize.
//! \param[in]      width     the desired width  of the image, in pixels.
//! \param[in]      height    the desired height of the image, in pixels.
//! \param[in]      format    the format of the pixels.
//! \param[in]      type      the type of the components of the pixels.
//! \param[in]      layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
//! \param[in]      memSpace  Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
//! \param[in]      alignment row byte alignment. Choose 0 or a power of 2.
//!                           1: yields no gap whatsoever between scanlines;
//!                           0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
//!                           Other common values are 16 or 32 for cache line size.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \return NVCV_ERR_MEMORY      if there is not enough memory to allocate the buffer.
NvCV_Status NvCV_API NvCVImage_Alloc(NvCVImage *im, unsigned width, unsigned height, NvCVImage_PixelFormat format,
  NvCVImage_ComponentType type, unsigned layout, unsigned memSpace, unsigned alignment);


//! Reallocate memory for, and initialize an image. This assumes that the image is valid.
//! It will check bufferBytes to see if enough memory is already available, and will reshape rather than realloc if true.
//! Otherwise, it will free the previous buffer and reallocate a new one.
//! \param[in,out]  im        the image to initialize.
//! \param[in]      width     the desired width  of the image, in pixels.
//! \param[in]      height    the desired height of the image, in pixels.
//! \param[in]      format    the format of the pixels.
//! \param[in]      type      the type of the components of the pixels.
//! \param[in]      layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
//! \param[in]      memSpace  Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
//! \param[in]      alignment row byte alignment. Choose 0 or a power of 2.
//!                           1: yields no gap whatsoever between scanlines;
//!                           0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
//!                           Other common values are 16 or 32 for cache line size.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \return NVCV_ERR_MEMORY      if there is not enough memory to allocate the buffer.
NvCV_Status NvCV_API NvCVImage_Realloc(NvCVImage *im, unsigned width, unsigned height, NvCVImage_PixelFormat format,
  NvCVImage_ComponentType type, unsigned layout, unsigned memSpace, unsigned alignment);


//! Deallocate the image buffer from the image. The image is not deallocated.
//! param[in,out] im  the image whose buffer is to be deallocated.
void NvCV_API NvCVImage_Dealloc(NvCVImage *im);


//! Allocate a new image, with storage (C-style constructor).
//! \param[in]      width     the desired width  of the image, in pixels.
//! \param[in]      height    the desired height of the image, in pixels.
//! \param[in]      format    the format of the pixels.
//! \param[in]      type      the type of the components of the pixels.
//! \param[in]      layout    One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.
//! \param[in]      memSpace  Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }
//! \param[in]      alignment row byte alignment. Choose 0 or a power of 2.
//!                           1: yields no gap whatsoever between scanlines;
//!                           0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
//!                           Other common values are 16 or 32 for cache line size.
//! \param[out]         *out will be a pointer to the new image if successful; otherwise NULL.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \return NVCV_ERR_MEMORY      if there is not enough memory to allocate the buffer.
NvCV_Status NvCV_API NvCVImage_Create(unsigned width, unsigned height, NvCVImage_PixelFormat format,
  NvCVImage_ComponentType type, unsigned layout, unsigned memSpace, unsigned alignment, NvCVImage **out);


//! Deallocate the image allocated with NvCVImage_Create() (C-style destructor).
void NvCV_API NvCVImage_Destroy(NvCVImage *im);


//! Get offsets for the components of a pixel format.
//! These are not byte offsets, but component offsets.
//! \param[in]  format  the pixel format to be interrogated.
//! \param[out] rOff    a place to store the offset for the red       channel (can be NULL).
//! \param[out] gOff    a place to store the offset for the green     channel (can be NULL).
//! \param[out] bOff    a place to store the offset for the blue      channel (can be NULL).
//! \param[out] aOff    a place to store the offset for the alpha     channel (can be NULL).
//! \param[out] yOff    a place to store the offset for the luminance channel (can be NULL).
void NvCV_API NvCVImage_ComponentOffsets(NvCVImage_PixelFormat format, int *rOff, int *gOff, int *bOff, int *aOff, int *yOff);


//! Transfer one image to another, with a limited set of conversions.
//!
//! If any of the images resides on the GPU, it may run asynchronously,
//! so cudaStreamSynchronize() should be called if it is necessary to run synchronously.
//! The following table indicates the currently-implemented conversions:
//!    +------------------+-------------+-------------+-------------+-------------+
//!    |                  |  u8 --> u8  |  u8 --> f32 | f32 --> u8  | f32 --> f32 |
//!    +------------------+-------------+-------------+-------------+-------------+
//!    | Y      -- > Y    |      X      |             |      X      |      X      |
//!    | Y      -- > A    |      X      |             |      X      |      X      |
//!    | Y      -- > RGB  |      X      |      X      |      X      |      X      |
//!    | Y      -- > RGBA |      X      |      X      |      X      |      X      |
//!    | A      -- > Y    |      X      |             |      X      |      X      |
//!    | A      -- > A    |      X      |             |      X      |      X      |
//!    | A      -- > RGB  |      X      |      X      |      X      |      X      |
//!    | A      -- > RGBA |      X      |             |             |             |
//!    | RGB    -- > Y    |      X      |      X      |             |             |
//!    | RGB    -- > A    |      X      |      X      |             |             |
//!    | RGB    -- > RGB  |      X      |      X      |      X      |      X      |
//!    | RGB    -- > RGBA |      X      |      X      |      X      |      X      |
//!    | RGBA   -- > Y    |      X      |      X      |             |             |
//!    | RGBA   -- > A    |             |      X      |             |             |
//!    | RGBA   -- > RGB  |      X      |      X      |      X      |      X      |
//!    | RGBA   -- > RGBA |      X      |             |             |             |
//!    | YUV420 -- > RGB  |      X      |             |             |             |
//!    | YUV422 -- > RGB  |      X      |             |             |             |
//!    +------------------+-------------+-------------+-------------+-------------+
//! where
//! * Either source or destination can be CHUNKY or PLANAR.
//! * Either source or destination can reside on the CPU or the GPU.
//! * The RGB components are in any order (i.e. RGB or BGR; RGBA or BGRA).
//! * YUV requires that the colorspace field be set manually prior to Transfer.
//! * Additionally, when the src and dst formats are the same, all formats are accommodated on CPU and GPU,
//! and this can be used as a replacement for cudaMemcpy2DAsync() (which it utilizes).
//!
//! When there is some kind of conversion AND the src and dst reside on different processors (CPU, GPU),
//! it is necessary to have a temporary GPU buffer, which is reshaped as needed to match the characteristics
//! of the CPU image. The same temporary image can be used in subsequent calls to NvCVImage_Transfer(),
//! regardless of the shape, format or component type, as it will grow as needed to accommodate
//! the largest memory requirement. The recommended usage for most cases is to supply an empty image
//! as the temporary; if it is not needed, no buffer is allocated. NULL can be supplied as the tmp
//! image, in which case an ephemeral buffer is allocated if needed, with resultant
//! performance degradation for image sequences.
//!
//! \param[in]      src     the source image.
//! \param[out]     dst     the destination image.
//! \param[in]      scale   a scale factor that can be applied when one (but not both) of the images
//!                         is based on floating-point components; this parameter is ignored when all image components
//!                         are represented with integer data types, or all image components are represented with
//!                         floating-point data types.
//! \param[in]      stream  the stream on which to perform the copy. This is ignored if both images reside on the CPU.
//! \param[in,out]  tmp     a temporary buffer that is sometimes needed when transferring images
//!                         between the CPU and GPU in either direction (can be empty or NULL).
//!                         It has the same characteristics as the CPU image, but it resides on the GPU.
//! \return         NVCV_SUCCESS           if successful,
//!                 NVCV_ERR_PIXELFORMAT   if one of the pixel formats is not accommodated.
//!                 NVCV_ERR_CUDA          if a CUDA error has occurred.
//!                 NVCV_ERR_GENERAL       if an otherwise unspecified error has occurred.
NvCV_Status NvCV_API NvCVImage_Transfer(
             const NvCVImage *src, NvCVImage *dst, float scale, struct CUstream_st *stream, NvCVImage *tmp);


//! Composite one BGRu8 source image over another using the given matte.
//! \param[in]  fg      the foreground source BGRu8 (or RGBu8) image.
//! \param[in]  bg      the background source BGRu8 (or RGBu8) image.
//! \param[in]  mat     the matte  Yu8   (or Au8)   image, indicating where the src should come through.
//! \param[out] dst     the destination BGRu8 (or RGBu8) image. This can be the same as fg or bg.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \bug    This is only implemented for 3-component u8 fg, bg and dst, and 1-component u8 mat,
//!         where all images are resident on the CPU.
NvCV_Status NvCV_API NvCVImage_Composite(const NvCVImage *fg, const NvCVImage *bg, const NvCVImage *mat, NvCVImage *dst);


//! Composite a BGRu8 source image over a constant color field using the given matte.
//! \param[in]      src     the source BGRu8 (or RGBu8) image.
//! \param[in]      mat     the matte  Yu8   (or Au8)   image, indicating where the src should come through.
//! \param[in]      bgColor the desired flat background color, with the same component ordering as the src and dst.
//! \param[in,out]  dst     the destination BGRu8 (or RGBu8) image. May be the same as src.
//! \return NVCV_SUCCESS         if the operation was successful.
//! \return NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
//! \bug    This is only implemented for 3-component u8 src and dst, and 1-component mat,
//!         where all images are resident on the CPU.
NvCV_Status NvCV_API NvCVImage_CompositeOverConstant(
              const NvCVImage *src, const NvCVImage *mat, const unsigned char bgColor[3], NvCVImage *dst);


//! Flip the image vertically.
//! No actual pixels are moved: it is just an accounting procedure.
//! This is extremely low overhead, but requires appropriate interpretation of the pitch.
//! Flipping twice yields the original orientation.
//! \param[in]  src  the source image (NULL implies src == dst).
//! \param[out] dst  the flipped image (can be the same as the src).
//! \return     NVCV_SUCCESS         if successful.
//! \return     NVCV_ERR_PIXELFORMAT for most planar formats.
//! \bug        This does not work for planar or semi-planar formats, neither RGB nor YUV.
//! \note       This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
NvCV_Status NvCV_API NvCVImage_FlipY(const NvCVImage *src, NvCVImage *dst);


//! Get the pointers for YUV, based on the format.
//! \param[in]  im          The image to be deconstructed.
//! \param[out] y           A place to store the pointer to y(0,0).
//! \param[out] u           A place to store the pointer to u(0,0).
//! \param[out] v           A place to store the pointer to v(0,0).
//! \param[out] yPixBytes   A place to store the byte stride between  luma  samples horizontally.
//! \param[out] cPixBytes   A place to store the byte stride between chroma samples horizontally.
//! \param[out] yRowBytes   A place to store the byte stride between  luma  samples vertically.
//! \param[out] cRowBytes   A place to store the byte stride between chroma samples vertically.
//! \return     NVCV_SUCCESS           If the information was gathered successfully.
//!             NVCV_ERR_PIXELFORMAT   Otherwise.
NvCV_Status NvCV_API NvCVImage_GetYUVPointers(NvCVImage *im,
  unsigned char **y, unsigned char **u, unsigned char **v,
  int *yPixBytes, int *cPixBytes, int *yRowBytes, int *cRowBytes);


#ifdef __cplusplus
} // extern "C"

/********************************************************************************
 * NvCVImage default constructor
 ********************************************************************************/

NvCVImage::NvCVImage() {
  pixels = nullptr;
  (void)NvCVImage_Alloc(this, 0, 0, NVCV_FORMAT_UNKNOWN, NVCV_TYPE_UNKNOWN, 0, 0, 0);
}

/********************************************************************************
 * NvCVImage allocation constructor
 ********************************************************************************/

NvCVImage::NvCVImage(unsigned width, unsigned height, NvCVImage_PixelFormat format, NvCVImage_ComponentType type,
                       unsigned layout, unsigned memSpace, unsigned alignment) {
  pixels = nullptr;
  (void)NvCVImage_Alloc(this, width, height, format, type, layout, memSpace, alignment);
}

/********************************************************************************
 * Subimage constructor
 ********************************************************************************/

NvCVImage::NvCVImage(NvCVImage *fullImg, int x, int y, unsigned width, unsigned height) {
  NvCVImage_InitView(this, fullImg, x, y, width, height);
}

/********************************************************************************
 * NvCVImage destructor
 ********************************************************************************/

NvCVImage::~NvCVImage() { NvCVImage_Dealloc(this); }

/********************************************************************************
 * copy subimage
 ********************************************************************************/

NvCV_Status NvCVImage::copyFrom(const NvCVImage *src, int srcX, int srcY, int dstX, int dstY, unsigned wd,
                                  unsigned ht) {
  NvCVImage srcView, dstView;
  NvCVImage_InitView(&srcView, const_cast<NvCVImage *>(src), srcX, srcY, wd, ht);
  NvCVImage_InitView(&dstView, this, dstX, dstY, wd, ht);
  return NvCVImage_Transfer(&srcView, &dstView, 1.f, 0, nullptr);
}

/********************************************************************************
 * copy image
 ********************************************************************************/

NvCV_Status NvCVImage::copyFrom(const NvCVImage *src) { return NvCVImage_Transfer(src, this, 1.f, 0, nullptr); }


#endif // ___cplusplus

#endif // __NVCVIMAGE_H__
