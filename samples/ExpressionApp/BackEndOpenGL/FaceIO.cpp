/*###############################################################################
#
# Copyright 2019-2021 NVIDIA Corporation
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

#include "FaceIO.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stack>
#include <string>
#include <vector>

#ifdef _MSC_VER
    #define strcasecmp  _stricmp
    #define strncasecmp _strnicmp
#endif /* _MSC_VER */

#define BAIL_IF_ERR(err)                    do { if ((err) != 0)          { goto bail;             } } while(0)
#define BAIL_IF_ZERO(x, err, code)          do { if ((x) == 0)            { err = code; goto bail; } } while(0)
#define BAIL_IF_NONZERO(x, err, code)       do { if ((x) != 0)            { err = code; goto bail; } } while(0)
#define BAIL_IF_FALSE(x, err, code)         do { if (!(x))                { err = code; goto bail; } } while(0)
#define BAIL_IF_TRUE(x, err, code)          do { if ((x))                 { err = code; goto bail; } } while(0)
#define BAIL_IF_NULL(x, err, code)          do { if ((void*)(x) == NULL)  { err = code; goto bail; } } while(0)
#define BAIL_IF_NEGATIVE(x, err, code)      do { if ((x) < 0)             { err = code; goto bail; } } while(0)
#define BAIL_IF_NONPOSITIVE(x, err, code)   do { if (!((x) > 0))          { err = code; goto bail; } } while(0)
#define BAIL(err, code)                     do {                            err = code; goto bail;   } while(0)

#ifndef __BYTE_ORDER__                  /* How bytes are packed into a 32 bit word */
 #define __ORDER_LITTLE_ENDIAN__ 3210   /* First byte in the least significant position */
 #define __ORDER_BIG_ENDIAN__    0123   /* First byte in the most  significant position */
 #ifdef _MSC_VER
  #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
 #endif /* _MSC_VER */
#endif /* __BYTE_ORDER__ */

#if       __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
 #define IOFOURCC(w, o, r, d)   (((w) << 24) | ((o) << 16) | ((r) << 8) | ((d) << 0))
#elif     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
 #define IOFOURCC(w, o, r, d)   (((d) << 24) | ((r) << 16) | ((o) << 8) | ((w) << 0))
#endif /* __BYTE_ORDER__ */

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                                  Utilities                           *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

/********************************************************************************
 * HasSuffix
 ********************************************************************************/

static bool HasSuffix(const char *str, const char *suf) {
  size_t  strSize = strlen(str),
          sufSize = strlen(suf);
  if (strSize < sufSize)
    return false;
  return (0 == strcasecmp(suf, str + strSize - sufSize));
}

/********************************************************************************
 * CopyDoubleToSingleVector
 * This works in-place.
 ********************************************************************************/

static void CopyDoubleToSingleVector(const double *d, float *s, uint32_t n) {
  for (float *send = s + n; s != send;) *s++ = (float)(*d++);
}

/********************************************************************************
 * CopyUInt32to16Vector
 * This works in-place.
 ********************************************************************************/

static void CopyUInt32to16Vector(const uint32_t *fr, uint16_t *to, uint32_t n) {
  for (uint16_t *toEnd = to + n; to != toEnd;) *to++ = (uint16_t)(*fr++);
}

/********************************************************************************
 * FaceIOErrorStringFromCode
 ********************************************************************************/

const char* FaceIOErrorStringFromCode(FaceIOErr err) {
  struct ErrStr { FaceIOErr err; const char *str; };
  static const ErrStr lut[] = {
    { kIOErrNone,         "no error"                                    },
    { kIOErrFileNotFound, "The file was not found"                      },
    { kIOErrFileOpen,     "The file could not be opened"                },
    { kIOErrEOF,          "A premature end-of-file was encountered"     },
    { kIOErrRead,         "An error occurred while reading a file"      },
    { kIOErrWrite,        "An error occurred while writing a file"      },
    { kIOErrSyntax,       "A parsing syntax error has been encountered" },
    { kIOErrFormat,       "The file has an unknown format"              },
    { kIOErrParameter,    "The parameter has an invalid value"          },
  };
  for (const ErrStr *p = lut; p != &lut[sizeof(lut)/sizeof(lut[0])]; ++p)
      if (p->err == err)
          return p->str;
  static char msg[16];
  snprintf(msg, sizeof(msg), "ERR#%d", (int)err);
  return msg;
}

/********************************************************************************
 * PrintIOError
 ********************************************************************************/

static void PrintIOError(const char *file, FaceIOErr err) {
  if (kIOErrNone == err) return;
  printf("\"%s\": %s\n", file, FaceIOErrorStringFromCode(err));
}

/********************************************************************************
 * PrintUnknownFormatMessage
 ********************************************************************************/

static void PrintUnknownFormatMessage(const char *file, const char *msg = nullptr) {
  if (!msg) msg = "";
  printf("\"%s\": Unknown format %s\n", file, msg);
}

/********************************************************************************
 * ReadFileIntoString
 ********************************************************************************/

static FaceIOErr ReadFileIntoString(const char *file, bool isText, std::string &str) {
  FaceIOErr err = kIOErrNone;
  FILE *fd = NULL;
  long z;
  size_t y;
  #ifndef _MSC_VER
    fd = fopen(file, (isText ? "r" : "rb"));
  #else  /* _MSC_VER */
    BAIL_IF_NONZERO(fopen_s(&fd, file, (isText ? "r" : "rb")), err, kIOErrFileNotFound);
  #endif /* _MSC_VER */
  BAIL_IF_NULL(fd, err, kIOErrFileNotFound);
  fseek(fd, 0L, SEEK_END);
  z = ftell(fd);
  BAIL_IF_NEGATIVE(z, err, kIOErrRead);   /* If there was a seek error, return an appropriate error code */
  fseek(fd, 0L, SEEK_SET);
  str.resize((unsigned long)z);
  y = fread(&str[0], 1, z, fd);    /* y differs from z when reading text on Windows, .. */
  str.resize(y);                          /* ... because CRLF --> NL */
bail:
  if (fd) fclose(fd);
  return err;
}

typedef struct IbugSfmMapper {
  unsigned numLandmarks;
  uint16_t landmarkMap[50][2];
  uint16_t rightContour[8];
  uint16_t leftContour[8];
}IbugSfmMapper;

const IbugSfmMapper ibugMapping = {
      68, 
      // Landmarks Mapping for 68 points
      {
                     // 1 to 8 are the right contour landmarks
        {9, 33},     // chin bottom
                     // 10 to 17 are the left contour landmarks
        {18, 225},   // right eyebrow outer-corner (18)
        {19, 229},   // right eyebrow between middle and outer corner
        {20, 233},   // right eyebrow middle, vertical middle (20)
        {21, 2086},  // right eyebrow between middle and inner corner
        {22, 157},   // right eyebrow inner-corner (19)
        {23, 590},   // left eyebrow inner-corner (23)
        {24, 2091},  // left eyebrow between inner corner and middle
        {25, 666},   // left eyebrow middle (24)
        {26, 662},   // left eyebrow between middle and outer corner
        {27, 658},   // left eyebrow outer-corner (22)
        {28, 2842},  // bridge of the nose (parallel to upper eye lids)
        {29, 379},   // middle of the nose, a bit below the lower eye lids
        {30, 272},   // above nose-tip (1cm or so)
        {31, 114},   // nose-tip (3)
        {32, 100},   // right nostril, below nose, nose-lip junction
        {33, 2794},  // nose-lip junction
        {34, 270},   // nose-lip junction (28)
        {35, 2797},  // nose-lip junction
        {36, 537},   // left nostril, below nose, nose-lip junction
        {37, 177},   // right eye outer-corner (1)
        {38, 172},   // right eye pupil top right (from subject's perspective)
        {39, 191},   // right eye pupil top left
        {40, 181},   // right eye inner-corner (5)
        {41, 173},   // right eye pupil bottom left
        {42, 174},   // right eye pupil bottom right
        {43, 614},   // left eye inner-corner (8)
        {44, 624},   // left eye pupil top right
        {45, 605},   // left eye pupil top left
        {46, 610},   // left eye outer-corner (2)
        {47, 607},   // left eye pupil bottom left
        {48, 606},   // left eye pupil bottom right
        {49, 398},   // right mouth corner (12)
        {50, 315},   // upper lip right top outer
        {51, 413},   // upper lip middle top right
        {52, 329},   // upper lip middle top (14)
        {53, 825},   // upper lip middle top left
        {54, 736},   // upper lip left top outer
        {55, 812},   // left mouth corner (13)
        {56, 841},   // lower lip left bottom outer
        {57, 693},   // lower lip middle bottom left
        {58, 411},   // lower lip middle bottom (17)
        {59, 264},   // lower lip middle bottom right
        {60, 431},   // lower lip right bottom outer
                     // 61 not defined - would be right inner corner of the mouth
        {62, 416},   // upper lip right bottom outer
        {63, 423},   // upper lip middle bottom
        {64, 828},   // upper lip left bottom outer
                     // 65 not defined - would be left inner corner of the mouth
        {66, 817},   // lower lip left top outer
        {67, 442},   // lower lip middle top
        {68, 404},   // lower lip right top outer
      },
      {1, 2, 3, 4, 5, 6, 7, 8}, // ibug right contour points
      {10, 11, 12, 13, 14, 15, 16, 17} // ibug left contour points
};

/********************************************************************************
 ********************************************************************************
 * NVF Encapsulated object definitions
 ********************************************************************************
 ********************************************************************************/

/********************************************************************************
 * Object encapsulation. The type is usually a FOURCC.
 ********************************************************************************/

struct EOTypeSize {
  uint32_t type, size;
};

/********************************************************************************
 * FOURCC types
 ********************************************************************************/

#define FOURCC_ADJACENT_FACES IOFOURCC('A', 'J', 'F', 'C')
#define FOURCC_ADJACENT_VERTICES IOFOURCC('A', 'J', 'V', 'X')
#define FOURCC_BASIS IOFOURCC('B', 'S', 'I', 'S')
#define FOURCC_BLEND_SHAPES IOFOURCC('B', 'L', 'N', 'D')
#define FOURCC_COLOR IOFOURCC('C', 'O', 'L', 'R')
#define FOURCC_EIGENVALUES IOFOURCC('E', 'I', 'V', 'L')
#define FOURCC_FILE_TYPE IOFOURCC('N', 'F', 'A', 'C')
#define FOURCC_IBUG IOFOURCC('I', 'B', 'U', 'G')
#define FOURCC_LANDMARK_MAP IOFOURCC('L', 'M', 'R', 'K')
#define FOURCC_LEFT_CONTOUR IOFOURCC('L', 'C', 'T', 'R')
#define FOURCC_MEAN IOFOURCC('M', 'E', 'A', 'N')
#define FOURCC_MODEL IOFOURCC('M', 'O', 'D', 'L')
#define FOURCC_MODEL_CONTOUR IOFOURCC('M', 'C', 'T', 'R')
#define FOURCC_NAME IOFOURCC('N', 'A', 'M', 'E')
#define FOURCC_RIGHT_CONTOUR IOFOURCC('R', 'C', 'T', 'R')
#define FOURCC_SHAPE IOFOURCC('S', 'H', 'A', 'P')
#define FOURCC_TEXTURE_COORDS IOFOURCC('T', 'X', 'C', 'O')
#define FOURCC_EOTOC IOFOURCC('T', 'O', 'C', '0')
#define FOURCC_TOPOLOGY IOFOURCC('T', 'O', 'P', 'O')
#define FOURCC_TRIANGLE_LIST IOFOURCC('T', 'R', 'N', 'G')
#define FOURCC_NVLM IOFOURCC('N', 'V', 'L', 'M')
#define FOURCC_PARTITIONS IOFOURCC('P', 'R', 'T', 'S')
#define FOURCC_PART IOFOURCC('P', 'A', 'R', 'T')
#define FOURCC_MATERIAL IOFOURCC('M', 'T', 'R', 'L')


/********************************************************************************
 *      NVFFileHeader
 ********************************************************************************/

struct NVFFileHeader {
  enum {                        // Illustrate 2 bit packets packed into a byte by the specified scheme
    LITTLE_ENDIAN_CODE = 0xE4,  // = 11 10 01 00    // or 0x10 = 0001 0000 -- 4 bit packets
    BIG_ENDIAN_CODE = 0x1B      // = 00 01 10 11    // or 0x01 = 0000 0001
  };
  unsigned type;
  unsigned size;
  unsigned char endian, sizeBits, indexBits, zero;
  unsigned tocLoc;

  NVFFileHeader() {
    type = FOURCC_FILE_TYPE;
    size = sizeof(*this) - sizeof(type) - sizeof(size);
    endian = LITTLE_ENDIAN_CODE;
    sizeBits = 32;   // 32 bits for the object size
    indexBits = 16;  // 16 bits for indices
    zero = 0;
    tocLoc = 0;
  }
  void clear() { memset(this, 0, sizeof(*this)); }
  static unsigned EOTOCOffset() {
    return (char *)(&((NVFFileHeader *)nullptr)->tocLoc) - (char *)(&((NVFFileHeader *)nullptr)->type);
  }
};
namespace { // anonymous

/********************************************************************************
 ********************************************************************************
 * EOWriter - Encapsulated Object Writer
 ********************************************************************************
 ********************************************************************************/

class EOWriter
{
public:

  EOWriter();
  ~EOWriter();

  /// Open a file for writing.
  /// @param[in]  fileName    the name of the file to be written to.
  /// @return     kIOErrNone   if the file was opened successfully.
  /// @return     kIOErrWrite  if there were problems opening the file for writing.
  FaceIOErr open(const char *fileName);

  /// Close the file without writing the table of contents, and flush the output.
  FaceIOErr close();

  /// Write the table of contents and close the file.
  /// @param[in]  tocRefOffset    the offset from the beginning of the file where the location of the TOC is stored.
  /// @return kIOErrNone   if the operation was completed successfully.
  FaceIOErr writeTocAndClose(unsigned tocRefOffset);

  /// Get the file descriptor.
  /// @return the file descriptor, or NULL if there is no open file.
  FILE* fileDescriptor();

  /// Write the object encapsulation header.
  /// @param[in]  type    a FOURCC 4-byte code.
  /// @param[in]  size    a 4-byte size for the remainder of the object.
  /// @param[in]  tag     a tag for the TOC. 0 makes no entry into the TOC.
  ///                     Usually the outer objects are in the TOC. But one may also include
  ///                     subobjects in the TOC, or no objects at all.
  /// @return kIOErrNone   if the operation was completed successfully.
  FaceIOErr writeEncapsulationHeader(unsigned type, unsigned size, unsigned tag = 0);

  /// Write an encapsulated opaque object.
  /// @param[in]  type    a FOURCC 4-byte code.
  /// @param[in]  size    the size in bytes of the data to be written.
  /// @param[in]  data    the data to be written.
  /// @param[in]  tag     a tag for the TOC. 0 makes no entry into the TOC.
  ///                     Usually the outer objects are in the TOC. But one may also include
  ///                     subobjects in the TOC, or no objects at all.
  /// @return kIOErrNone   if the operation was completed successfully.
  FaceIOErr writeOpaqueObject(unsigned type, unsigned size, const void *data, unsigned tag = 0);
  FaceIOErr writeOpaqueObject(unsigned type, size_t   size, const void *data, unsigned tag = 0)
          { return writeOpaqueObject(type, (unsigned)size, data, tag); }

  /// Write opaque data
  /// @param[in]  size    the size in bytes of the data to be written.
  /// @param[in]  data    the data to be written.
  /// @return kIOErrNone   if the operation was completed successfully.
  FaceIOErr writeData(unsigned size, const void *data);
  FaceIOErr writeData(size_t   size, const void *data) { return writeData((unsigned)size, data); }

private:
  struct Impl;
  Impl *pimpl;
};

struct EOTOC {
  unsigned _tag;
  unsigned _offset;
  EOTOC(unsigned tag, unsigned offset) {
    _tag = tag;
    _offset = offset;
  }
};

struct EOEncapsulation {
  unsigned _type, _size;
  EOEncapsulation(unsigned type, unsigned size) {
    _type = type;
    _size = size;
  }
};

struct EOWriter::Impl {
  FILE *fd;
  std::vector<EOTOC> toc;

  void addToEOTOC(unsigned tag) {
    if (tag) toc.emplace_back(tag, (unsigned)ftell(fd));
  }
};

EOWriter::EOWriter() {
  pimpl = new EOWriter::Impl;
  pimpl->fd = nullptr;
}

EOWriter::~EOWriter() {
  if (pimpl->fd) fclose(pimpl->fd);
  delete pimpl;
}

FILE *EOWriter::fileDescriptor() { return pimpl->fd; }

FaceIOErr EOWriter::open(const char *fileName) {
  pimpl->fd = nullptr;
  #ifndef _MSC_VER
    pimpl->fd = fopen(fileName, "wb");
  #else  /* _MSC_VER */
    if (0 != fopen_s(&pimpl->fd, fileName, "wb"))
      pimpl->fd = nullptr;
  #endif /* _MSC_VER */
  return pimpl->fd ? kIOErrNone : kIOErrWrite;
}

FaceIOErr EOWriter::close() {
  fclose(pimpl->fd);
  pimpl->fd = nullptr;
  return kIOErrNone;
}

FaceIOErr EOWriter::writeEncapsulationHeader(unsigned type, unsigned size, unsigned tag) {
  EOEncapsulation eo(type, size);
  pimpl->addToEOTOC(tag);
  size_t z = fwrite(&eo, sizeof(eo._type), sizeof(eo) / sizeof(eo._type), pimpl->fd);
  return (z == (sizeof(eo) / sizeof(eo._type))) ? kIOErrNone : kIOErrWrite;
}

FaceIOErr EOWriter::writeData(unsigned size, const void *data) {
  return (1 == fwrite(data, size, 1, pimpl->fd)) ? kIOErrNone : kIOErrWrite;
}

FaceIOErr EOWriter::writeOpaqueObject(unsigned type, unsigned size, const void *data, unsigned tag) {
  FaceIOErr err = writeEncapsulationHeader(type, size, tag);
  if (!err) err = writeData(size, data);
  return err;
}

FaceIOErr EOWriter::writeTocAndClose(unsigned tocRefOffset) {
  FaceIOErr err = kIOErrNone;
  unsigned numRecords = (unsigned)pimpl->toc.size();
  if (numRecords) {
    unsigned offset = ftell(pimpl->fd),
             size   = sizeof(unsigned) + numRecords * sizeof(EOTOC);
    err = writeEncapsulationHeader(FOURCC_EOTOC, size, 0);
    if (kIOErrNone == err) {
      unsigned recordSize = sizeof(EOTOC);
      BAIL_IF_ERR(
          err = (1 == fwrite(&recordSize, sizeof(recordSize), 1, pimpl->fd))
                    ? kIOErrNone
                    : kIOErrWrite);
      BAIL_IF_ERR(err = (numRecords == fwrite(pimpl->toc.data(), sizeof(EOTOC),
                                              numRecords, pimpl->fd))
                            ? kIOErrNone
                            : kIOErrWrite);
      BAIL_IF_NONZERO(fseek(pimpl->fd, tocRefOffset, SEEK_SET), err,
                      kIOErrWrite);
      BAIL_IF_ERR(err = (1 == fwrite(&offset, sizeof(offset), 1, pimpl->fd))
                            ? kIOErrNone
                            : kIOErrWrite);
    }
  }
  fclose(pimpl->fd);
  pimpl->fd = nullptr;
bail:
  return err;
}

/********************************************************************************
 ********************************************************************************
 * JSON reader and writer
 ********************************************************************************
 ********************************************************************************/

enum JSONNodeType {
  kJSONObject,
  kJSONArray,
  kJSONString,
  kJSONNumber,
  kJSONMember,
  kJSONBoolean,
  kJSONNull
};

struct JSONInfo {
  void *userData;
  JSONNodeType type;
  const char *value;
  double number;
};

typedef FaceIOErr (*JSONOpenNodeProc)(JSONInfo *info);
typedef FaceIOErr (*JSONCloseNodeProc)(JSONInfo *info);

class JSONReader {
 public:
  JSONReader(JSONOpenNodeProc openNode, JSONCloseNodeProc closeNode);
  ~JSONReader() {}

  /// Parse from a string.
  /// @param[in]      str     the string to be parsed.
  /// @param[in]      len     the length of the string to be parsed; if 0, it is
  /// assumed to be NULL-terminated.
  /// @param[in,out]  state   the parser state.
  /// @return     kIOErrNone   if the operation was completed successfully.
  /// @return     kIOErrSyntax if a syntax error occurred during parsing.
  /// @return     kIOErrEOF    if the file ended sooner than expected.
  FaceIOErr parse(const char *str, size_t len, void *state);

  /// Parse from a file.
  /// @param[in]      fileName    the file to be parsed.
  /// @param[in,out]  state       the parser state.
  /// @return     kIOErrNone   if the operation was completed successfully.
  /// @return     kIOErrSyntax if a syntax error occurred during parsing.
  /// @return     kIOErrEOF    if the file ended sooner than expected.
  FaceIOErr parse(const char *fileName, void *state);

 private:
  JSONOpenNodeProc _openNode;
  JSONCloseNodeProc _closeNode;
  const char *_jsonStr;
  size_t _jsonLen;
  JSONInfo _infoBack;
  void skipWhiteSpace();
  FaceIOErr readNumber();
  FaceIOErr readString();
  FaceIOErr readValue();
  FaceIOErr readArray();
  FaceIOErr readObject();
  FaceIOErr getString(std::string *str);
};

class JSONWriter {
 public:
  JSONWriter();
  ~JSONWriter();
  FaceIOErr open(const char *file);
  FaceIOErr close();
  void openObject(const char *tag = nullptr);
  void closeObject();
  void openArray(const char *tag = nullptr);
  void closeArray();
  void writeNumericArray(unsigned n, const float *v, unsigned maxRow = 0, const char *tag = nullptr);
  void writeNumericArray(unsigned n, const double *v, unsigned maxRow = 0, const char *tag = nullptr);
  void writeNumericArray(unsigned n, const int *v, unsigned maxRow = 0, const char *tag = nullptr);
  void writeNumericArray(unsigned n, const unsigned short *v, unsigned maxRow = 0, const char *tag = nullptr);
  void writeNumericArray(unsigned n, const unsigned char *v, unsigned maxRow = 0, const char *tag = nullptr);
  void writeNumber(double number, const char *tag = nullptr);
  void writeString(const char *str, const char *tag = nullptr);
  void writeBool(bool value, const char *tag = nullptr);
  void writeNull(const char *tag = nullptr);
 private:
  class Impl;
  Impl *pimpl;
};


JSONReader::JSONReader(JSONOpenNodeProc openNode, JSONCloseNodeProc closeNode) {
  _openNode          = openNode;
  _closeNode         = closeNode;
  _jsonStr           = nullptr;
  _jsonLen           = 0;
  _infoBack.userData = nullptr;
  _infoBack.type     = kJSONNull;
  _infoBack.value    = nullptr;
  _infoBack.number   = NAN;
}

void JSONReader::skipWhiteSpace() {
  for (; _jsonLen && isspace(*_jsonStr); ++_jsonStr, --_jsonLen) {}
}

FaceIOErr JSONReader::getString(std::string *str) {
  FaceIOErr err = kIOErrNone;
  const char *s;
  char c, quote;
  size_t n;

  str->clear();
  BAIL_IF_FALSE(_jsonLen > 2, err, kIOErrEOF);               /* not enough characters */
  n = _jsonLen;
  c = *(s = _jsonStr);
  BAIL_IF_FALSE(('"' == c || '\'' == c), err, kIOErrSyntax); /* missing quotes */
  quote = c;
  for (--n, ++s; n--; s++)
    if (quote == *s)                                         /* TODO: check for escapes */
      break;
  n = s - _jsonStr + 1;
  BAIL_IF_FALSE(n >= 2, err, kIOErrEOF);                     /* not enough characters */
  str->assign(_jsonStr + 1, n - 2);
  _jsonStr += n;
  _jsonLen -= n;

bail:
  if (kIOErrSyntax == err) printf("Expecting a string, but found no quotes\n");
  return err;
}

FaceIOErr JSONReader::readString() {
  FaceIOErr err;
  std::string str;

  BAIL_IF_ERR(err = getString(&str));
  _infoBack.type = kJSONString;
  _infoBack.value = str.c_str();
  BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
  err = (*_closeNode)(&_infoBack);

bail:
  _infoBack.value = nullptr;                                /* Make Coverity happy */
  return err;
}

FaceIOErr JSONReader::readObject() {
  FaceIOErr err;
  std::string str;

  _infoBack.type = kJSONObject;                              /* Open an object node */
  _infoBack.value = nullptr;
  BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
  BAIL_IF_FALSE('{' == _jsonStr[0], err, kIOErrSyntax);      /* This is where we always start */
  --_jsonLen;
  BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
  ++_jsonStr;
  skipWhiteSpace();

  if ('}' != _jsonStr[0]) {
    do {
      skipWhiteSpace();
      BAIL_IF_ERR(err = getString(&str));                   /* Every member must have a name */
      skipWhiteSpace();
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      BAIL_IF_FALSE(':' == _jsonStr[0], err, kIOErrSyntax); /* Followed by a colon */
      --_jsonLen;
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      ++_jsonStr;
      skipWhiteSpace();
      _infoBack.type = kJSONMember;                         /* Start a new member */
      _infoBack.value = str.c_str();
      BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
      BAIL_IF_ERR(err = readValue());                       /* Get the member */
      _infoBack.type = kJSONMember;                         /* Close the member */
      BAIL_IF_ERR(err = (*_closeNode)(&_infoBack));
      skipWhiteSpace();
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      if ('}' == _jsonStr[0])                               /* Look for ending */
        break;                                              /* Normal object exit */
      BAIL_IF_FALSE(',' == _jsonStr[0], err, kIOErrNotValue); /* Comma-separated members */
      --_jsonLen;
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      ++_jsonStr;
    } while (1);
  }

  --_jsonLen;                                               /* Always ('}' == _jsonStr[0]) here */
  ++_jsonStr;
  _infoBack.type = kJSONObject;                             /* Close the object node */
  _infoBack.value = nullptr;
  err = (*_closeNode)(&_infoBack);

bail:
  _infoBack.value = nullptr;                                /* Make Coverity happy */
  return err;
}

FaceIOErr JSONReader::readArray() {
  FaceIOErr err;

  _infoBack.type = kJSONArray;
  _infoBack.value = nullptr;
  BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
  BAIL_IF_FALSE('[' == _jsonStr[0], err, kIOErrSyntax); /* We always enter in this state */
  --_jsonLen;
  BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
  ++_jsonStr;
  skipWhiteSpace();

  if (']' != _jsonStr[0]) {
    do {
      skipWhiteSpace();
      BAIL_IF_ERR(err = readValue());
      skipWhiteSpace();
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      if (']' == _jsonStr[0]) break;
      BAIL_IF_FALSE(',' == _jsonStr[0], err, kIOErrEOF);
      --_jsonLen;
      BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);
      ++_jsonStr;
    } while (1);
  }

  --_jsonLen;                                         /* Always (']' == _jsonStr[0]) here */
  ++_jsonStr;
  _infoBack.type = kJSONArray;
  _infoBack.value = nullptr;
  err = (*_closeNode)(&_infoBack);

bail:
  return err;
}

FaceIOErr JSONReader::readNumber() {
  FaceIOErr err = kIOErrNone;
  char *endPtr;
  size_t n;
  std::string str;

  _infoBack.number = strtod(_jsonStr, &endPtr); /* Parse the number */
  n = endPtr - _jsonStr;
  BAIL_IF_FALSE(n <= _jsonLen, err, kIOErrEOF);
  BAIL_IF_ZERO(n, err, kIOErrSyntax);
  str.assign(_jsonStr, n);
  _jsonStr += n;
  _infoBack.type = kJSONNumber;               /* Open bracket */
  _infoBack.value = str.c_str();
  BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
  err = (*_closeNode)(&_infoBack);            /* Close bracket */

bail:
  _infoBack.value = nullptr;                  /* Make Coverity happy */
  return err;
}

FaceIOErr JSONReader::readValue() {
  static const char kTrue[] = "true", kFalse[] = "false", kNull[] = "null";
  FaceIOErr err = kIOErrNotValue;

  skipWhiteSpace();
  BAIL_IF_ZERO(_jsonLen, err, kIOErrEOF);

  if ('{' == _jsonStr[0])                                 // Object
    return readObject();

  if ('[' == _jsonStr[0])                                 // Array
    return readArray();

  if ('"' == _jsonStr[0] || '\'' == _jsonStr[0])          // String
    return readString();

  if (('0' <= _jsonStr[0] && _jsonStr[0] <= '9') ||       // Number
      '-' == _jsonStr[0] || '+' == _jsonStr[0] || '.' == _jsonStr[0])
    return readNumber();

  if (_jsonLen >= 4 && !strncasecmp(_jsonStr, kTrue, 4))  // Boolean true
  { // TODO: look out for "true" token prefix
    _jsonStr += 4;
    _jsonLen -= 4;
    _infoBack.type = kJSONBoolean;
    _infoBack.value = kTrue;
    BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
    return (*_closeNode)(&_infoBack);
  }

  if (_jsonLen >= 5 && !strncasecmp(_jsonStr, kFalse, 5))  // Boolean false
  {  // TODO: look out for "false" token prefix
    _jsonStr += 5;
    _jsonLen -= 5;
    _infoBack.type = kJSONBoolean;
    _infoBack.value = kFalse;
    BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
    return (*_closeNode)(&_infoBack);
  }

  if (_jsonLen >= 4 && !strncasecmp(_jsonStr, kNull, 4))  // null
  {  // TODO: look out for " null" token prefix
    _jsonStr += 4;
    _jsonLen -= 4;
    _infoBack.type = kJSONNull;
    _infoBack.value = kNull;
    BAIL_IF_ERR(err = (*_openNode)(&_infoBack));
    return (*_closeNode)(&_infoBack);
  }

bail:
  return err;
}

FaceIOErr JSONReader::parse(const char *str, size_t len, void *userState) {
  FaceIOErr err;

  _jsonStr = str;
  _jsonLen = len ? len : strlen(str);
  _infoBack.userData = userState;
  err = readValue();
  if (kIOErrNone != err) printf("Syntax error at \"%.32s...\"\n", _jsonStr);
  return err;
}

FaceIOErr JSONReader::parse(const char *fileName, void *userState) {
  std::string str;
  FaceIOErr err = ReadFileIntoString(fileName, true, str);
  if (kIOErrNone == err) err = parse(str.c_str(), str.size(), userState);
  return err;
}


class JSONWriter::Impl {
 public:
  int level;
  FILE *fd;
  std::stack<int> count;
  void doIndent();
  void maybeComma();
  void maybeTag(const char *tag);
  void commaIndentTag(const char *tag);
  template<typename T> void writeArray(unsigned n, const T *v, unsigned maxRow, const char *fmt, const char *tag);
  void openObject(const char *tag);
  void closeObject();
  void openArray(const char *tag);
  void closeArray();
};

void JSONWriter::Impl::doIndent() {
  for (int i = level * 2; i-- > 0;) putc(' ', fd);
}

void JSONWriter::Impl::maybeComma() {
  if (count.top())
    putc(',', fd);
  count.top()++;
}

void JSONWriter::Impl::maybeTag(const char *tag) {
  if (!tag) return;
  fprintf(fd, "\"%s\": ", tag);
}

void JSONWriter::Impl::commaIndentTag(const char *tag) {
  maybeComma();
  if (level) putc('\n', fd);
  doIndent();
  maybeTag(tag);
}
void JSONWriter::Impl::openObject(const char *tag)
{
  commaIndentTag(tag);
  putc('{', fd);
  ++(level);
  count.push(0);
}

void JSONWriter::Impl::closeObject()
{
  --(level);
  putc('\n', fd);
  doIndent();
  putc('}', fd);
  count.pop();
}

void JSONWriter::Impl::openArray(const char *tag)
{
  commaIndentTag(tag);
  putc('[', fd);
  ++(level);
  count.push(0);
}

void JSONWriter::Impl::closeArray()
{
  --(level);
  putc('\n', fd);
  doIndent();
  putc(']', fd);
  count.pop();
}

JSONWriter::JSONWriter() {
  pimpl = new Impl();
  pimpl->level = 0;
  pimpl->fd = nullptr;
  pimpl->count.push(0);
}

JSONWriter::~JSONWriter() {
  if (pimpl->fd && pimpl->fd != stdout) fclose(pimpl->fd);
  delete pimpl;
}

FaceIOErr JSONWriter::open(const char *file) {
  if (pimpl->fd && pimpl->fd != stdout) (void)(this->close());
  if (file && file[0]) {
    #ifndef _MSC_VER
      pimpl->fd = fopen(file, "w");
    #else  /* _MSC_VER */
      if (0 != fopen_s(&pimpl->fd, file, "w"))
        pimpl->fd = nullptr;
    #endif /* _MSC_VER */
  } else {
    pimpl->fd = stdout;
  }
  return pimpl->fd ? kIOErrNone : kIOErrWrite;
}

FaceIOErr JSONWriter::close() {
  putc('\n', pimpl->fd);
  if (pimpl->fd && pimpl->fd != stdout) (void)fclose(pimpl->fd);
  pimpl->fd = nullptr;
  return kIOErrNone;
}

void JSONWriter::openObject(const char *tag) {
  pimpl->openObject(tag);
}

void JSONWriter::closeObject() {
  pimpl->closeObject();
}

void JSONWriter::openArray(const char *tag) {
  pimpl->openArray(tag);
}

void JSONWriter::closeArray() {
  pimpl->closeArray();
}

template<typename T>
void JSONWriter::Impl::writeArray(unsigned n, const T * v, unsigned maxRow, const char *fmt, const char * tag)
{
  openArray(tag);
  putc('\n', fd);
  if (n) {
    for (; n > maxRow; n -= maxRow) {
      doIndent();
      for (unsigned i = maxRow; i--; ++v) {
        fprintf(fd, fmt, *v, (i ? ' ' : '\n'));
        putc(',', fd);
        putc((i ? ' ' : '\n'), fd);
      }
    }
    doIndent();
    for (; n--; ++v) {
      fprintf(fd, fmt, *v, (n ? ", " : ""));
      if (n)
        fprintf(fd, ", ");
    }
  }
  closeArray();
}


void JSONWriter::writeNumericArray(unsigned n, const float *v, unsigned maxRow, const char *tag) {
  pimpl->writeArray(n, v, maxRow, "%.8g", tag);
}

void JSONWriter::writeNumericArray(unsigned n, const double *v, unsigned maxRow, const char *tag) {
  pimpl->writeArray(n, v, maxRow, "%.17g", tag);
}

void JSONWriter::writeNumericArray(unsigned n, const int *v, unsigned maxRow, const char *tag) {
  pimpl->writeArray(n, v, maxRow, "%d", tag);
}

void JSONWriter::writeNumericArray(unsigned n, const unsigned short *v, unsigned maxRow, const char *tag) {
  pimpl->writeArray(n, v, maxRow, "%u", tag);
}

void JSONWriter::writeNumericArray(unsigned n, const unsigned char *v, unsigned maxRow, const char *tag) {
  pimpl->writeArray(n, v, maxRow, "%u", tag);
}

void JSONWriter::writeNumber(double number, const char *tag) {
  pimpl->commaIndentTag(tag);
  fprintf(pimpl->fd, "%.17g", number);
}

void JSONWriter::writeString(const char *str, const char *tag) {
  pimpl->commaIndentTag(tag);
  fprintf(pimpl->fd, "\"%s\"", str);
}

void JSONWriter::writeBool(bool value, const char *tag) {
  pimpl->commaIndentTag(tag);
  fprintf(pimpl->fd, "%s", value ? "true" : "false");
}

void JSONWriter::writeNull(const char *tag) {
  pimpl->commaIndentTag(tag);
  fprintf(pimpl->fd, "null");
}


} // namespace anonymous


/*******************************************************************************
********************************************************************************
********************************************************************************
*****                               NVF Output                             *****
********************************************************************************
********************************************************************************
********************************************************************************/

namespace /* anonymous */ {

template <typename ElementType>
uint32_t NVFSize(uint32_t n, const ElementType *v) {
  return (uint32_t)(n * sizeof(*v));
}

template <typename ElementType>
FaceIOErr NVFWriteOpaqueObject(uint32_t type, uint32_t size, const ElementType *v, EOWriter &wtr, unsigned tag = 0) {
  return wtr.writeOpaqueObject(type, size * sizeof(*v), v, tag);
}

}  // namespace

static uint32_t NVFSizeShapeModel(const FaceIOAdapter *fac) {
  uint32_t size = 8 + NVFSize(fac->getShapeMeanSize(), fac->getShapeMean())
                + 8 + NVFSize(fac->getShapeModesSize(), fac->getShapeModes())
                + sizeof(uint32_t)
                + 8 + NVFSize(fac->getShapeEigenvaluesSize(), fac->getShapeEigenvalues())
                + 8 + NVFSize(fac->getTriangleListSize(), fac->getTriangleList());
  return size;
}

static FaceIOErr NVFWriteShapeModel(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  uint32_t numModes = fac->getShapeNumModes(), sizeNumMode = sizeof(uint32_t),
           sizeMean = NVFSize(fac->getShapeMeanSize(), fac->getShapeMean()),
           sizeModes = NVFSize(fac->getShapeModesSize(), fac->getShapeModes()),
           sizeEigen = NVFSize(fac->getShapeEigenvaluesSize(), fac->getShapeEigenvalues()),
           sizeTriList = NVFSize(fac->getTriangleListSize(), fac->getTriangleList()),
           size = 8 + sizeMean + 8 + sizeModes + sizeNumMode + 8 + sizeEigen + 8 + sizeTriList;
  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_SHAPE, size, 0));
  BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_MEAN, sizeMean, fac->getShapeMean(), 0));
  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_BASIS, sizeModes + sizeNumMode, 0));
  BAIL_IF_ERR(err = wtr.writeData(sizeNumMode, &numModes));
  BAIL_IF_ERR(err = wtr.writeData(sizeModes, fac->getShapeModes()));
  BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_EIGENVALUES, sizeEigen, fac->getShapeEigenvalues(), 0));
  BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_TRIANGLE_LIST, sizeTriList, fac->getTriangleList(), 0));
bail:
  return err;
}

static uint32_t NVFSizeColorModel(const FaceIOAdapter *fac) {
  uint32_t sizeMean = NVFSize(fac->getColorMeanSize(), fac->getColorMean()),
           sizeModes = NVFSize(fac->getColorModesSize(), fac->getColorModes()),
           sizeEigen = NVFSize(fac->getColorEigenvaluesSize(), fac->getColorEigenvalues()),
           sizeTriList = 0,  // NVFSize(fac->getTriangleListSize(), fac->getTriangleList());
      size = sizeMean + sizeModes + sizeEigen + sizeTriList;
  if (size) size += 8 + 8 + 8 + 8;
  return size;
}

static FaceIOErr NVFWriteColorModel(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err = kIOErrNone;
  uint32_t numModes = fac->getColorNumModes(), sizeNumMode = sizeof(uint32_t),
           sizeMean = NVFSize(fac->getColorMeanSize(), fac->getColorMean()),
           sizeModes = NVFSize(fac->getColorModesSize(), fac->getColorModes()),
           sizeEigen = NVFSize(fac->getColorEigenvaluesSize(), fac->getColorEigenvalues()),
           sizeTriList = 0,  // NVFSize(fac->getTriangleListSize(), fac->getTriangleList());
      size = 8 + sizeMean + 8 + sizeModes + sizeNumMode + 8 + sizeEigen + 8 + sizeTriList;
  if (sizeMean + sizeModes + sizeEigen != 0) {
    BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_COLOR, size, 0));
    BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_MEAN, sizeMean, fac->getColorMean(), 0));
    BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_BASIS, sizeModes + sizeNumMode, 0));
    BAIL_IF_ERR(err = wtr.writeData(sizeNumMode, &numModes));
    BAIL_IF_ERR(err = wtr.writeData(sizeModes, fac->getColorModes()));
    BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_EIGENVALUES, sizeEigen, fac->getColorEigenvalues(), 0));
    BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_TRIANGLE_LIST, sizeTriList, fac->getTriangleList(), 0));
  }
bail:
  return err;
}

static uint32_t NVFSizeMorphableModel(const FaceIOAdapter *fac) {
  uint32_t  size        = 8 + NVFSizeShapeModel(fac),
            textureSize = NVFSize(fac->getTextureCoordinatesSize(), fac->getTextureCoordinates()),
            colorSize   = NVFSizeColorModel(fac);
  if (textureSize) size += 8 + textureSize;
  if (colorSize)   size += 8 + colorSize;
  return size;
}

static FaceIOErr NVFWriteMorphableModel(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  uint32_t size;

  /* Write the shape model */
  BAIL_IF_ERR(err = NVFWriteShapeModel(fac, wtr));

  /* Write the color model */
  BAIL_IF_ERR(err = NVFWriteColorModel(fac, wtr));

  /* Write the texture coordinates */
  if (0 != (size = NVFSize(fac->getTextureCoordinatesSize(), fac->getTextureCoordinates())))
    BAIL_IF_ERR(err = wtr.writeOpaqueObject(FOURCC_TEXTURE_COORDS, size, fac->getTextureCoordinates()));

bail:
  return err;
}

static uint32_t NVFSizeString(const char *str) {
  return (strlen(str) + 3) & ~3;  // bump up to multiple of 4
}

static FaceIOErr NVFWriteString(const char *str, EOWriter &wtr) {
  uint32_t strSize = (uint32_t)strlen(str);
  FaceIOErr err = wtr.writeData(strSize, str);
  uint32_t pad = (uint32_t)(NVFSizeString(str) - strSize);
  BAIL_IF_ERR(err);
  if (pad) err = wtr.writeData(pad, "\0\0\0");
bail:
  return err;
}

static uint32_t NVFSizeBlendShapes(const FaceIOAdapter *fac) {
  uint32_t size = sizeof(uint32_t),  // sizeof(numShapes)
      numShapes = fac->getNumBlendShapes(), i;
  for (i = 0; i < numShapes; ++i) {
    size += 8 + NVFSizeString(fac->getBlendShapeName(i));
    size += 8 + NVFSize(fac->getBlendShapeSize(i), fac->getBlendShape(i));
  }
  return size;
}

static FaceIOErr NVFWriteBlendShapes(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err = kIOErrNone;
  uint32_t numShapes = fac->getNumBlendShapes();
  uint32_t i;

  BAIL_IF_ERR(err = wtr.writeData(sizeof(numShapes), &numShapes));
  for (i = 0; i < numShapes; ++i) {
    const char *str = fac->getBlendShapeName(i);
    BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_NAME, NVFSizeString(str)));
    BAIL_IF_ERR(err = NVFWriteString(str, wtr));
    BAIL_IF_ERR(err = NVFWriteOpaqueObject(FOURCC_SHAPE, fac->getBlendShapeSize(i), fac->getBlendShape(i), wtr));
  }
bail:
  return err;
}

static uint32_t NVFSizeIbugMappings(const FaceIOAdapter *fac) {
  uint32_t size;
  size = 8 + NVFSize(fac->getIbugLandmarkMappingsSize(), fac->getIbugLandmarkMappings())
       + 8 + NVFSize(fac->getIbugRightContourSize(),     fac->getIbugRightContour())
       + 8 + NVFSize(fac->getIbugLeftContourSize(),      fac->getIbugLeftContour());
  return size;
}

static FaceIOErr NVFWriteIbugMappings(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  BAIL_IF_ERR(err = NVFWriteOpaqueObject(FOURCC_LANDMARK_MAP, fac->getIbugLandmarkMappingsSize(),
                                          fac->getIbugLandmarkMappings(), wtr));
  BAIL_IF_ERR(err = NVFWriteOpaqueObject(FOURCC_RIGHT_CONTOUR, fac->getIbugRightContourSize(),
                                          fac->getIbugRightContour(), wtr));
  BAIL_IF_ERR(
      err = NVFWriteOpaqueObject(FOURCC_LEFT_CONTOUR, fac->getIbugLeftContourSize(), fac->getIbugLeftContour(), wtr));
bail:
  return err;
}

static uint32_t NVFSizeModelContours(const FaceIOAdapter *fac) {
  uint32_t size = 8 + NVFSize(fac->getModelRightContourSize(), fac->getModelRightContour())
                + 8 + NVFSize(fac->getModelLeftContourSize(), fac->getModelLeftContour());
  return size;
}

static FaceIOErr NVFWriteModelContours(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  BAIL_IF_ERR(err = NVFWriteOpaqueObject(FOURCC_RIGHT_CONTOUR, fac->getModelRightContourSize(),
                                         fac->getModelRightContour(), wtr));
  BAIL_IF_ERR(
      err = NVFWriteOpaqueObject(FOURCC_LEFT_CONTOUR, fac->getModelLeftContourSize(), fac->getModelLeftContour(), wtr));
bail:
  return err;
}

static uint32_t NVFSizeTopology(const FaceIOAdapter *fac) {
  uint32_t size = 8 + NVFSize(fac->getAdjacentFacesSize(), fac->getAdjacentFaces()) + 8 +
                  NVFSize(fac->getAdjacentVerticesSize(), fac->getAdjacentVertices());
  return size;
}

static FaceIOErr NVFWriteTopology(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  BAIL_IF_ERR(
      err = NVFWriteOpaqueObject(FOURCC_ADJACENT_FACES, fac->getAdjacentFacesSize(), fac->getAdjacentFaces(), wtr));
  BAIL_IF_ERR(err = NVFWriteOpaqueObject(FOURCC_ADJACENT_VERTICES, fac->getAdjacentVerticesSize(),
                                         fac->getAdjacentVertices(), wtr));
bail:
  return err;
}


static uint32_t NVFSizeNVLM(const FaceIOAdapter *fac) {
  uint32_t size = 8 + NVFSize(fac->getNvlmLandmarksSize(),    fac->getNvlmLandmarks())
                + 8 + NVFSize(fac->getNvlmRightContourSize(), fac->getNvlmRightContour())
                + 8 + NVFSize(fac->getNvlmLeftContourSize(),  fac->getNvlmLeftContour());
  return size;
}

static FaceIOErr NVFWriteNVLM(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err;
  BAIL_IF_ERR(
    err = NVFWriteOpaqueObject(FOURCC_LANDMARK_MAP, fac->getNvlmLandmarksSize(), fac->getNvlmLandmarks(), wtr));
  BAIL_IF_ERR(
    err = NVFWriteOpaqueObject(FOURCC_RIGHT_CONTOUR, fac->getNvlmRightContourSize(), fac->getNvlmRightContour(), wtr));
  BAIL_IF_ERR(
    err = NVFWriteOpaqueObject(FOURCC_LEFT_CONTOUR, fac->getNvlmLeftContourSize(), fac->getNvlmLeftContour(), wtr));

bail:
  return err;
}

struct TPart {
  uint32_t partitionIndex, faceIndex, numFaces, vertexIndex, numVertices;
  int32_t smoothingGroup;
};

static uint32_t NVFSizePartitions(const FaceIOAdapter *fac) {
  uint32_t  numPartitions = fac->getNumPartitions(),
            size          = sizeof(numPartitions),
            i, z;
  for (i = 0; i < numPartitions; ++i) {
    size += 8;  // type, size
    size += sizeof(TPart); // partitionIndex, faceIndex, numFaces, vertexIndex, numVertices, smoothingGroup
    if (0 != (z = NVFSizeString(fac->getPartitionName(i))))         size += 8 + z;
    if (0 != (z = NVFSizeString(fac->getPartitionMaterialName(i)))) size += 8 + z;
  }
  return size;
}


static FaceIOErr NVFWritePartitions(const FaceIOAdapter *fac, EOWriter &wtr) {
  FaceIOErr err = kIOErrNone;
  uint32_t numPartitions = fac->getNumPartitions(), i;
  EOTypeSize ts;
  TPart part;
  const char *name, *mtrl;
  uint32_t nameSize, mtrlSize;

  ts.type = FOURCC_PART;
  BAIL_IF_ERR(err = wtr.writeData(sizeof(numPartitions), &numPartitions));

  for (i = 0; i < numPartitions; ++i) {
    part.partitionIndex = fac->getPartition(
        i, &part.faceIndex, &part.numFaces, &part.vertexIndex, &part.numVertices, &part.smoothingGroup);
    nameSize = (nullptr != (name = fac->getPartitionName(i))         && name[0]) ? NVFSizeString(name) : 0;
    mtrlSize = (nullptr != (mtrl = fac->getPartitionMaterialName(i)) && mtrl[0]) ? NVFSizeString(mtrl) : 0;
    ts.size = sizeof(part);
    if (nameSize) ts.size += 8 + nameSize;
    if (mtrlSize) ts.size += 8 + mtrlSize;
    BAIL_IF_ERR(err = wtr.writeData(sizeof(ts), &ts));      // {type,size}
    BAIL_IF_ERR(err = wtr.writeData(sizeof(part), &part));  // unencapsulated 6 int16s
    if (nameSize) {
      BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_NAME, nameSize));
      BAIL_IF_ERR(err = NVFWriteString(name, wtr));
    }
    if (mtrlSize) {
      BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_MATERIAL, mtrlSize));
      BAIL_IF_ERR(err = NVFWriteString(mtrl, wtr));
    }
  }

bail:
  return err;
}


/********************************************************************************
 * API                          WriteNVFFaceModel                           API *
 ********************************************************************************/

FaceIOErr WriteNVFFaceModel(FaceIOAdapter *fac, const char *fileName) {
  FaceIOErr err;
  EOWriter wtr;
  NVFFileHeader header;

  err = wtr.open(fileName);
  if (kIOErrNone != err) {
    PrintIOError(fileName, err);
    goto bail;
  }

  BAIL_IF_ERR(err = wtr.writeData(sizeof(header), &header));

  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_MODEL, NVFSizeMorphableModel(fac), FOURCC_MODEL));
  BAIL_IF_ERR(err = NVFWriteMorphableModel(fac, wtr));

  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_IBUG, NVFSizeIbugMappings(fac), FOURCC_IBUG));
  BAIL_IF_ERR(err = NVFWriteIbugMappings(fac, wtr));

  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_BLEND_SHAPES, NVFSizeBlendShapes(fac), FOURCC_BLEND_SHAPES));
  BAIL_IF_ERR(err = NVFWriteBlendShapes(fac, wtr));

  BAIL_IF_ERR(err =
                  wtr.writeEncapsulationHeader(FOURCC_MODEL_CONTOUR, NVFSizeModelContours(fac), FOURCC_MODEL_CONTOUR));
  BAIL_IF_ERR(err = NVFWriteModelContours(fac, wtr));

  BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_TOPOLOGY, NVFSizeTopology(fac), FOURCC_TOPOLOGY));
  BAIL_IF_ERR(err = NVFWriteTopology(fac, wtr));

  if (fac->getNvlmLandmarksSize()) {
    BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_NVLM, NVFSizeNVLM(fac), FOURCC_NVLM));
    BAIL_IF_ERR(err = NVFWriteNVLM(fac, wtr));
  }

  if (fac->getNumPartitions()) {
    BAIL_IF_ERR(err = wtr.writeEncapsulationHeader(FOURCC_PARTITIONS, NVFSizePartitions(fac), FOURCC_PARTITIONS));
    BAIL_IF_ERR(err = NVFWritePartitions(fac, wtr));
  }

  BAIL_IF_ERR(err = wtr.writeTocAndClose(NVFFileHeader::EOTOCOffset()));

bail:
  return err;
}

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                               NVF Input                              *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/


static size_t SafeRead(void *ptr, size_t size, size_t num, FILE *fd) {
  if (ptr) return(fread(ptr, size, num, fd));
  else     return fseek(fd, (long)(size * num), SEEK_CUR) ? 0 : num;
}

static FaceIOErr NVFReadShapeModel(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n, numModes;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_MEAN:
        n = ts.size / sizeof(*fac->getShapeMean());
        BAIL_IF_FALSE(1 == SafeRead(fac->getShapeMean(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_BASIS:
        BAIL_IF_FALSE((1 == fread(&numModes, sizeof(numModes), 1, fd)), numModes, 0);  // The number of modes
        ts.size -= sizeof(numModes);                                                   // The byte size of all modes
        n = ts.size / sizeof(*fac->getShapeModes());                                   // The elements size of all nodes
        BAIL_IF_FALSE(1 == SafeRead(fac->getShapeModes(n / numModes, numModes), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_EIGENVALUES:
        n = ts.size / sizeof(*fac->getShapeEigenvalues());
        BAIL_IF_FALSE(1 == SafeRead(fac->getShapeEigenvalues(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_TRIANGLE_LIST:
        n = ts.size / sizeof(*fac->getTriangleList());
        BAIL_IF_FALSE(1 == SafeRead(fac->getTriangleList(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadColorModel(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n, numModes;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_MEAN:
        n = ts.size / sizeof(*fac->getColorMean());
        BAIL_IF_FALSE(1 == SafeRead(fac->getColorMean(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_BASIS:
        BAIL_IF_FALSE((1 == fread(&numModes, sizeof(numModes), 1, fd)), numModes, 0);  // The number of modes
        ts.size -= sizeof(numModes);                                                   // The byte size of all modes
        n = ts.size / sizeof(*fac->getColorModes());                                   // The elements size of all nodes
        BAIL_IF_FALSE(1 == SafeRead(fac->getColorModes(n / numModes, numModes), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_EIGENVALUES:
        n = ts.size / sizeof(*fac->getColorEigenvalues());
        BAIL_IF_FALSE(1 == SafeRead(fac->getColorEigenvalues(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_TRIANGLE_LIST:
        (void)fseek(fd, ts.size, SEEK_CUR);  // Skip color triangle list -- it doesn't make sense
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadMorphableModel(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_SHAPE:
        BAIL_IF_ERR(err = NVFReadShapeModel(fac, ts.size, fd));
        break;
      case FOURCC_COLOR:
        BAIL_IF_ERR(err = NVFReadColorModel(fac, ts.size, fd));
        break;
      case FOURCC_TEXTURE_COORDS:
        n = ts.size / sizeof(*fac->getTextureCoordinates());
        BAIL_IF_FALSE(1 == SafeRead(fac->getTextureCoordinates(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadString(std::string &str, uint32_t size, FILE *fd) {
  str.resize(size);
  FaceIOErr err = (size == fread(&str[0], 1, size, fd)) ? kIOErrNone : kIOErrRead;
  BAIL_IF_ERR(err);
  for (; size; --size)
    if (str[size - 1]) break;
  str.resize(size);  // remove pad
bail:
  return err;
}

static FaceIOErr NVFReadBlendShapes(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  uint32_t numShapes;
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  std::string name;
  float *shape;

  BAIL_IF_FALSE(1 == fread(&numShapes, sizeof(numShapes), 1, fd), err, kIOErrEOF);
  fac->setNumBlendShapes(numShapes);

  for (uint32_t idxShape = 0, idxName = 0; size >= sizeof(ts);) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_NAME:
        BAIL_IF_FALSE(idxName < numShapes, err, kIOErrRead);
        BAIL_IF_ERR(err = NVFReadString(name, ts.size, fd));
        fac->setBlendShapeName(idxName++, name.c_str());
        break;
      case FOURCC_SHAPE:
        BAIL_IF_FALSE(idxShape < numShapes, err, kIOErrRead);
        shape = fac->getBlendShape(idxShape++, ts.size / sizeof(*shape));
        BAIL_IF_FALSE(1 == SafeRead(shape, ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
    return err;
}

static FaceIOErr NVFReadIbugMappings(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_LANDMARK_MAP:
        n = ts.size / sizeof(*fac->getIbugLandmarkMappings());
        BAIL_IF_FALSE(1 == SafeRead(fac->getIbugLandmarkMappings(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_RIGHT_CONTOUR:
        n = ts.size / sizeof(*fac->getIbugRightContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getIbugRightContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_LEFT_CONTOUR:
        n = ts.size / sizeof(*fac->getIbugLeftContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getIbugLeftContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadModelContours(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  ;
  EOTypeSize ts;
  uint32_t n;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_RIGHT_CONTOUR:
        n = ts.size / sizeof(*fac->getModelRightContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getModelRightContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_LEFT_CONTOUR:
        n = ts.size / sizeof(*fac->getModelLeftContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getModelLeftContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadTopology(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_ADJACENT_FACES:
        n = ts.size / sizeof(*fac->getAdjacentFaces());
        BAIL_IF_FALSE(1 == SafeRead(fac->getAdjacentFaces(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_ADJACENT_VERTICES:
        n = ts.size / sizeof(*fac->getAdjacentVertices());
        BAIL_IF_FALSE(1 == SafeRead(fac->getAdjacentVertices(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadNvlm(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t n;

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_LANDMARK_MAP:
        n = ts.size / sizeof(*fac->getNvlmLandmarks());
        BAIL_IF_FALSE(1 == SafeRead(fac->getNvlmLandmarks(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_RIGHT_CONTOUR:
        n = ts.size / sizeof(*fac->getIbugRightContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getNvlmRightContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      case FOURCC_LEFT_CONTOUR:
        n = ts.size / sizeof(*fac->getIbugLeftContour());
        BAIL_IF_FALSE(1 == SafeRead(fac->getNvlmLeftContour(n), ts.size, 1, fd), err, kIOErrRead);
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
bail:
  return err;
}

static FaceIOErr NVFReadPart(uint32_t i, FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  TPart part;
  std::string name;

  BAIL_IF_FALSE(sizeof(part) <= size, err, kIOErrEOF);
  BAIL_IF_FALSE(1 == fread(&part, sizeof(part), 1, fd), err, kIOErrEOF);
  size -= sizeof(part);
  fac->setPartition(i, part.faceIndex, part.numFaces, part.vertexIndex, part.numVertices, part.smoothingGroup);
  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_NAME:
        BAIL_IF_ERR(err = NVFReadString(name, ts.size, fd));
        fac->setPartitionName(i, name.c_str());
        break;
      case FOURCC_MATERIAL:
        BAIL_IF_ERR(err = NVFReadString(name, ts.size, fd));
        fac->setPartitionMaterialName(i, name.c_str());
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
  BAIL_IF_NONZERO(size, err, kIOErrRead); // We are out-of-sync if size != 0
bail:
  return err;
}

static FaceIOErr NVFReadPartitions(FaceIOAdapter *fac, uint32_t size, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  EOTypeSize ts;
  uint32_t numPartitions, partIx = 0;

  BAIL_IF_FALSE(sizeof(numPartitions) <= size, err, kIOErrEOF);
  BAIL_IF_FALSE(1 == fread(&numPartitions, sizeof(numPartitions), 1, fd), err, kIOErrEOF);
  size -= sizeof(numPartitions);
  fac->setNumPartitions(numPartitions);

  for (; size >= sizeof(ts); ++partIx) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_PART:
        BAIL_IF_ERR(err = NVFReadPart(partIx, fac, ts.size, fd));
        break;
      default:
        BAIL_IF_NONZERO(fseek(fd, ts.size, SEEK_CUR), err, kIOErrEOF);  // We skip over objects we don't understand
        break;
    }
  }
  BAIL_IF_NONZERO(size, err, kIOErrRead); // We are out-of-sync if size != 0
bail:
  return err;
}

/********************************************************************************
 * API                              ReadNVFFaceModel                        API *
 ********************************************************************************/

FaceIOErr ReadNVFFaceModel(const char *fileName, FaceIOAdapter *fac) {
  FaceIOErr err = kIOErrNone;
  FILE *fd = nullptr;
  EOTypeSize ts;
  NVFFileHeader header;
  uint32_t size;

#ifndef _MSC_VER
  fd = fopen(fileName, "rb");
#else  /* _MSC_VER */
  if (0 != fopen_s(&fd, fileName, "rb"))
    fd = nullptr;
#endif /* _MSC_VER */
  if (!fd) {
    PrintIOError(fileName, err = kIOErrFileOpen);
    goto bail;
  }

  /* Find the length of the file */
  BAIL_IF_NEGATIVE(fseek(fd, 0L, SEEK_END), err, kIOErrRead);
  size = ftell(fd);
  BAIL_IF_NEGATIVE(fseek(fd, 0L, SEEK_SET), err, kIOErrRead);
  BAIL_IF_FALSE(size >= sizeof(header), err, kIOErrEOF);

  /* Validate header */
  BAIL_IF_FALSE(1 == fread(&header, sizeof(header), 1, fd), err, kIOErrRead);
  BAIL_IF_FALSE(FOURCC_FILE_TYPE == header.type, err, kIOErrFormat);
  BAIL_IF_FALSE(8 <= header.size, err, kIOErrFormat);                                    // TODO: robustify
  BAIL_IF_FALSE(NVFFileHeader::LITTLE_ENDIAN_CODE == header.endian, err, kIOErrFormat);  // We only handle little-endian
  BAIL_IF_FALSE(32 == header.sizeBits, err, kIOErrFormat);                               // We only use 32 bit sizes
  BAIL_IF_FALSE(16 == header.indexBits, err, kIOErrFormat);                              // We only use 16 bit indices
  size -= sizeof(header);
  if (header.size > 8) { /* Forward compatibility */
    uint32_t extra = header.size - 8;
    BAIL_IF_FALSE(size >= extra, err, kIOErrEOF);
    size -= extra;
    BAIL_IF_NEGATIVE(fseek(fd, extra, SEEK_CUR), err, kIOErrRead);
  }

  while (size >= sizeof(ts)) {
    BAIL_IF_FALSE(1 == fread(&ts, sizeof(ts), 1, fd), err, kIOErrEOF);
    size -= sizeof(ts);
    BAIL_IF_FALSE(ts.size <= size, err, kIOErrEOF);
    size -= ts.size;
    switch (ts.type) {
      case FOURCC_MODEL:
        BAIL_IF_ERR(err = NVFReadMorphableModel(fac, ts.size, fd));
        break;
      case FOURCC_IBUG:
        BAIL_IF_ERR(err = NVFReadIbugMappings(fac, ts.size, fd));
        break;
      case FOURCC_BLEND_SHAPES:
        BAIL_IF_ERR(err = NVFReadBlendShapes(fac, ts.size, fd));
        break;
      case FOURCC_MODEL_CONTOUR:
        BAIL_IF_ERR(err = NVFReadModelContours(fac, ts.size, fd));
        break;
      case FOURCC_TOPOLOGY:
        BAIL_IF_ERR(err = NVFReadTopology(fac, ts.size, fd));
        break;
      case FOURCC_NVLM:
        BAIL_IF_ERR(err = NVFReadNvlm(fac, ts.size, fd));
        break;
      case FOURCC_PARTITIONS:
        BAIL_IF_ERR(err = NVFReadPartitions(fac, ts.size, fd));
        break;
      case FOURCC_EOTOC:
        BAIL_IF_NEGATIVE(fseek(fd, ts.size, SEEK_CUR), err, kIOErrRead);  // Skip the EOTOC
        break;
      default:
        BAIL(err, kIOErrSyntax);  // TODO: We should just skip over objects we don't understand
    }
  }

bail:
  if (fd) fclose(fd);
  return err;
}

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                               EOS Input                              *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

#define DEBUG_PARSER 0

static FaceIOErr EOSReadModesSize(uint32_t *oneModeSize, uint32_t *numModes, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  BAIL_IF_FALSE(1 == fread(oneModeSize, sizeof(*oneModeSize), 1, fd), err, kIOErrRead);
  BAIL_IF_FALSE(1 == fread(numModes, sizeof(*numModes), 1, fd), err, kIOErrRead);
bail:
  return err;
}

static FaceIOErr EOSReadShapeModel(FaceIOAdapter *fac, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  uint32_t modeSize, numModes;
  uint16_t *tri;
  float *data;
  long long z;

  if (DEBUG_PARSER) printf("Reading Shape Model @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  if (DEBUG_PARSER) printf("Reading mean @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));  // Coverity thinks "read" means "taint"
  if (0 != (modeSize *= numModes)) {
    BAIL_IF_NULL(data = fac->getShapeMean(modeSize), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Reading pca_basis @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
  if (0 != (modeSize * numModes)) {
    BAIL_IF_NULL(data = fac->getShapeModes(modeSize, numModes), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * numModes * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Reading eigenvalues @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
  if (modeSize *= numModes) {
    BAIL_IF_NULL(data = fac->getShapeEigenvalues(modeSize), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Reading triangle_list(%lld) @ %ld (%#lx)\n", z, ftell(fd), ftell(fd));
  BAIL_IF_FALSE(1 == fread(&z, sizeof(z), 1, fd), err, kIOErrRead);
  z *= 3;
  tri = fac->getTriangleList((uint32_t)z * 2);  // Big enough for uint32_t now, we resize to uint16_t later
  BAIL_IF_FALSE(z == (long long)fread(tri, sizeof(uint32_t), z, fd), err, kIOErrRead);
  CopyUInt32to16Vector((uint32_t *)tri, tri, (uint32_t)z);
  fac->setTriangleListSize((uint32_t)z);  // Now it is the actual size

bail:
  if (err) printf("Error reading Shape Model\n");
  return err;
}

static FaceIOErr EOSReadColorModel(FaceIOAdapter *fac, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  uint32_t modeSize, numModes;
  float *data;
  long long z;

  if (DEBUG_PARSER) printf("Reading Color Model @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  if (DEBUG_PARSER) printf("Reading mean @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
  if (modeSize *= numModes) {
    BAIL_IF_NULL(data = fac->getColorMean(modeSize), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Reading pca_basis @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
  if (modeSize * numModes) {
    BAIL_IF_NULL(data = fac->getColorModes(modeSize, numModes), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * numModes * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Reading eigenvalues @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
  if (modeSize *= numModes) {
    BAIL_IF_NULL(data = fac->getColorEigenvalues(modeSize), err, kIOErrNullPointer);
    BAIL_IF_FALSE(1 == fread(data, modeSize * sizeof(*data), 1, fd), err, kIOErrRead);
  }

  if (DEBUG_PARSER) printf("Skipping triangle_list(%lld) @ %ld (%#lx)\n", z, ftell(fd), ftell(fd));
  BAIL_IF_FALSE(1 == fread(&z, sizeof(z), 1, fd), err, kIOErrRead);
  z *= 3 * sizeof(uint32_t);
  (void)fseek(fd, (long)z, SEEK_CUR);

bail:
  if (err) printf("Error reading Color Model\n");
  return err;
}

static FaceIOErr EOSReadMorphableModel(FaceIOAdapter *fac, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  int version;
  size_t n1;
  long long z;
  float *tex;

  if (DEBUG_PARSER) printf("Reading TmorphableModel @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  n1 = fread(&version, sizeof(version), 1, fd);
  BAIL_IF_FALSE(n1 == 1, err, kIOErrRead);
  if (DEBUG_PARSER) printf("Version(%d)\n", version);
  if (DEBUG_PARSER) printf("Reading shape_model @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadShapeModel(fac, fd));
  if (DEBUG_PARSER) printf("Reading color_model @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_ERR(err = EOSReadColorModel(fac, fd));
  BAIL_IF_FALSE(1 == fread(&z, sizeof(z), 1, fd), err, kIOErrRead);
  z *= 2;                                            /* 2 floats per texture coordinate */
  tex = fac->getTextureCoordinates((uint32_t)z * 2); /* allocate enough space for double precision */
  if (DEBUG_PARSER) printf("Reading texture_coordinates(%lld) @ %ld (%#lx)\n", z, ftell(fd), ftell(fd));
  BAIL_IF_FALSE(1 == fread(tex, sizeof(double) * z, 1, fd), err, kIOErrRead);
  CopyDoubleToSingleVector((double *)tex, tex, (uint32_t)z);
  fac->setTextureCoordinatesSize((uint32_t)z); /* resize for single precision */
bail:
  if (err) printf("Error reading TmorphableModel\n");
  return err;
}

static FaceIOErr EOSReadString(std::string &str, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  long long z;
  if (DEBUG_PARSER) printf("Reading string @ %ld (%#lx)\n", ftell(fd), ftell(fd));
  BAIL_IF_FALSE(1 == fread(&z, sizeof(z), 1, fd), err, kIOErrRead);
  str.resize(z);
  BAIL_IF_FALSE(z == (long long)fread(&str[0], sizeof(str[0]), z, fd), err, kIOErrRead);
bail:
  if (err) printf("Error reading EOS string\n");
  return err;
}

static FaceIOErr EOSReadBlendShapes(FaceIOAdapter *fac, FILE *fd) {
  FaceIOErr err = kIOErrNone;
  size_t numShapes;
  uint32_t i, modeSize, numModes;
  float *modes;
  long long z;
  std::string name;

  BAIL_IF_FALSE(1 == fread(&z, sizeof(z), 1, fd), err, kIOErrRead);
  numShapes = z;
  fac->setNumBlendShapes((uint32_t)numShapes);
  for (i = 0; i < (uint32_t)numShapes; ++i) {
    BAIL_IF_ERR(err = EOSReadString(name, fd));
    fac->setBlendShapeName(i, name.c_str());
    BAIL_IF_ERR(err = EOSReadModesSize(&modeSize, &numModes, fd));
    modes = fac->getBlendShape(i, modeSize *= numModes);
    BAIL_IF_FALSE(1 == fread(modes, modeSize * sizeof(*modes), 1, fd), err, kIOErrRead);
  }

bail:
  if (err) printf("Error reading TblendShapes\n");
  return err;
}

struct EOSContoursReaderState {
  enum {
    STATE_NULL,
    STATE_MODEL_CONTOUR,
    STATE_RIGHT_CONTOUR,
    STATE_LEFT_CONTOUR,
    STATE_RIGHT_CONTOUR_ARRAY,
    STATE_LEFT_CONTOUR_ARRAY,
    STATE_ERROR
  };
  int state, nest;
  FaceIOAdapter *fac;
  EOSContoursReaderState() {
    state = STATE_NULL;
    nest = 0;
    fac = nullptr;
  };
};

static FaceIOErr EOSContoursOpenNode(JSONInfo *info) {
  EOSContoursReaderState *st = (EOSContoursReaderState *)(info->userData);
  switch (info->type) {
    case kJSONObject:
      ++(st->nest);
      break;
    case kJSONArray:
      ++(st->nest);
      switch (st->state) {
        case EOSContoursReaderState::STATE_RIGHT_CONTOUR:
          st->state = EOSContoursReaderState::STATE_RIGHT_CONTOUR_ARRAY;
          break;
        case EOSContoursReaderState::STATE_LEFT_CONTOUR:
          st->state = EOSContoursReaderState::STATE_LEFT_CONTOUR_ARRAY;
          break;
        default:
          st->state = EOSContoursReaderState::STATE_ERROR;
          break;
      }
      break;
    case kJSONNumber:
      switch (st->state) {
        case EOSContoursReaderState::STATE_RIGHT_CONTOUR_ARRAY:
          st->fac->appendModelRightContour((uint16_t)info->number);
          break;
        case EOSContoursReaderState::STATE_LEFT_CONTOUR_ARRAY:
          st->fac->appendModelLeftContour((uint16_t)info->number);
          break;
        default:
          st->state = EOSContoursReaderState::STATE_ERROR;
          break;
      }
      break;
    case kJSONMember:
      if (!strcmp(info->value, "model_contour"))
        st->state = EOSContoursReaderState::STATE_MODEL_CONTOUR;
      else if (!strcmp(info->value, "right_contour"))
        st->state = EOSContoursReaderState::STATE_RIGHT_CONTOUR;
      else if (!strcmp(info->value, "left_contour"))
        st->state = EOSContoursReaderState::STATE_LEFT_CONTOUR;
      else
        st->state = EOSContoursReaderState::STATE_ERROR;
      break;
  }
  return (EOSContoursReaderState::STATE_ERROR == st->state) ? kIOErrSyntax : kIOErrNone;
}

static FaceIOErr EOSContoursCloseNode(JSONInfo *info) {
  EOSContoursReaderState *st = (EOSContoursReaderState *)(info->userData);
  switch (info->type) {
    case kJSONObject:
      --(st->nest);
      break;
    case kJSONArray:
      --(st->nest);
      st->state = EOSContoursReaderState::STATE_NULL;
      break;
  }
  return kIOErrNone;
}

static FaceIOErr EOSReadContours(FaceIOAdapter *fac, const char *fileName) {
  JSONReader rdr(&EOSContoursOpenNode, &EOSContoursCloseNode);
  EOSContoursReaderState st;
  FaceIOErr err;

  fac->setModelRightContourSize(0);
  fac->setModelLeftContourSize(0);
  st.fac = fac;
  err = rdr.parse(fileName, &st);
  return err;
}

struct EOSTopologyReaderState {
  enum {
    STATE_NULL,
    STATE_EDGES,
    STATE_FACES,
    STATE_VERTICES,
    STATE_FACES_ARRAY,
    STATE_VERTICES_ARRAY,
    STATE_FACES_ARRAY_OBJ,
    STATE_VERTICES_ARRAY_OBJ,
    STATE_ERROR
  };
  int state, nest;
  FaceIOAdapter *fac;
  EOSTopologyReaderState() {
    state = STATE_NULL;
    nest = 0;
    fac = nullptr;
  };
};

static FaceIOErr EOSTopologyOpenNode(JSONInfo *info) {
  EOSTopologyReaderState *st = (EOSTopologyReaderState *)(info->userData);
  switch (info->type) {
    case kJSONObject:
      ++(st->nest);
      switch (st->state) {
        case EOSTopologyReaderState::STATE_FACES_ARRAY:
          st->state = EOSTopologyReaderState::STATE_FACES_ARRAY_OBJ;
          break;
        case EOSTopologyReaderState::STATE_VERTICES_ARRAY:
          st->state = EOSTopologyReaderState::STATE_VERTICES_ARRAY_OBJ;
          break;
        default:
          break;
      }
      break;
    case kJSONArray:
      ++(st->nest);
      switch (st->state) {
        case EOSTopologyReaderState::STATE_FACES:
          st->state = EOSTopologyReaderState::STATE_FACES_ARRAY;
          break;
        case EOSTopologyReaderState::STATE_VERTICES:
          st->state = EOSTopologyReaderState::STATE_VERTICES_ARRAY;
          break;
        default:
          st->state = EOSTopologyReaderState::STATE_ERROR;
          break;
      }
      break;
    case kJSONNumber:
      switch (st->state) {
        case EOSTopologyReaderState::STATE_FACES_ARRAY:
        case EOSTopologyReaderState::STATE_FACES_ARRAY_OBJ:
          st->fac->appendAdjacentFace((uint16_t)info->number);
          break;
        case EOSTopologyReaderState::STATE_VERTICES_ARRAY:
        case EOSTopologyReaderState::STATE_VERTICES_ARRAY_OBJ:
          st->fac->appendAdjacentVertex((uint16_t)info->number);
          break;
        default:
          st->state = EOSTopologyReaderState::STATE_ERROR;
          break;
      }
      break;
    case kJSONMember:
      if (!strcmp(info->value, "edge_topology"))
        st->state = EOSTopologyReaderState::STATE_EDGES;
      else if (!strcmp(info->value, "adjacent_faces"))
        st->state = EOSTopologyReaderState::STATE_FACES;
      else if (!strcmp(info->value, "adjacent_vertices"))
        st->state = EOSTopologyReaderState::STATE_VERTICES;
      else if ((!strcmp(info->value, "value0") || !strcmp(info->value, "value1")) &&
               (EOSTopologyReaderState::STATE_FACES_ARRAY_OBJ == st->state ||
                EOSTopologyReaderState::STATE_VERTICES_ARRAY_OBJ == st->state)) {
      } else
        st->state = EOSTopologyReaderState::STATE_ERROR;
      break;
  }
  return (EOSTopologyReaderState::STATE_ERROR == st->state) ? kIOErrSyntax : kIOErrNone;
}

static FaceIOErr EOSTopologyCloseNode(JSONInfo *info) {
  EOSTopologyReaderState *st = (EOSTopologyReaderState *)(info->userData);
  switch (info->type) {
    case kJSONObject:
      --(st->nest);
      switch (st->state) {
        case EOSTopologyReaderState::STATE_FACES_ARRAY_OBJ:
          st->state = EOSTopologyReaderState::STATE_FACES_ARRAY;
          break;
        case EOSTopologyReaderState::STATE_VERTICES_ARRAY_OBJ:
          st->state = EOSTopologyReaderState::STATE_VERTICES_ARRAY;
          break;
        default:
          st->state = EOSTopologyReaderState::STATE_ERROR;
          break;
      }
      break;
    case kJSONArray:
      --(st->nest);
      st->state = EOSTopologyReaderState::STATE_NULL;
      break;
  }
  return kIOErrNone;
}

static FaceIOErr EOSReadTopology(FaceIOAdapter *fac, const char *fileName) {
  JSONReader rdr(&EOSTopologyOpenNode, &EOSTopologyCloseNode);
  EOSTopologyReaderState st;
  FaceIOErr err;

  fac->setAdjacentFacesSize(0);
  fac->setAdjacentVerticesSize(0);
  st.fac = fac;
  err = rdr.parse(fileName, &st);
  return err;
}

/********************************************************************************
 * API                              ReadEOSFaceModel                        API *
 ********************************************************************************/

FaceIOErr ReadEOSFaceModel(const char *shape, unsigned int /*ibugNumLandmarks*/, const char *blendShapes, const char *contours,
                       const char *topology, FaceIOAdapter *fac) {
  FaceIOErr err = kIOErrNone;
  FILE *fd;

  if (shape) /* Shape */
  {
    if (!HasSuffix(shape, ".bin")) {
      PrintUnknownFormatMessage(shape, "for shape");
      BAIL(err, kIOErrFormat);
    }
#ifndef _MSC_VER
    fd = fopen(shape, "rb");
#else  /* _MSC_VER */
    if (0 != fopen_s(&fd, shape, "rb"))
      fd = nullptr;
#endif /* _MSC_VER */
    if (!fd) {
      PrintIOError(shape, err = kIOErrFileOpen);
      goto bail;
    }
    err = EOSReadMorphableModel(fac, fd);
    fclose(fd);
    BAIL_IF_ERR(err);
  }

  if (0) /* IBUG */
  {
    /* Insert ibug mappings reading code here */
  } else {

    unsigned n = sizeof(ibugMapping.landmarkMap) / sizeof(ibugMapping.landmarkMap[0][0]);
    memcpy(fac->getIbugLandmarkMappings(n), &ibugMapping.landmarkMap[0][0],
           n * sizeof(*fac->getIbugLandmarkMappings()));
    n = sizeof(ibugMapping.rightContour) / sizeof(ibugMapping.rightContour[0]);
    memcpy(fac->getIbugRightContour(n), ibugMapping.rightContour, n * sizeof(*fac->getIbugRightContour()));
    n = sizeof(ibugMapping.leftContour) / sizeof(ibugMapping.leftContour[0]);
    memcpy(fac->getIbugLeftContour(n), ibugMapping.leftContour, n * sizeof(*fac->getIbugLeftContour()));
  }

  if (blendShapes) /* Blend Shapes */
  {
    if (!HasSuffix(blendShapes, ".bin")) {
      PrintUnknownFormatMessage(blendShapes, "for blend shapes");
      BAIL(err, kIOErrFormat);
    }
#ifndef _MSC_VER
    fd = fopen(blendShapes, "rb");
#else  /* _MSC_VER */
    if (0 != fopen_s(&fd, blendShapes, "rb"))
      fd = nullptr;
#endif /* _MSC_VER */
    if (!fd) {
      PrintIOError(blendShapes, err = kIOErrFileOpen);
      goto bail;
    }
    err = EOSReadBlendShapes(fac, fd);
    fclose(fd);
    BAIL_IF_ERR(err);
  }

  if (contours) /* Contours */
  {
    if (!HasSuffix(contours, ".json")) {
      PrintUnknownFormatMessage(contours, "for contours");
      BAIL(err, kIOErrFormat);
    }
    err = EOSReadContours(fac, contours);
    if (kIOErrNone != err) {
      PrintIOError(contours, err);
      goto bail;
    }
  }

  if (topology) /* Topology */
  {
    if (!HasSuffix(topology, ".json")) {
      PrintUnknownFormatMessage(topology, "for topology");
      BAIL(err, kIOErrFormat);
    }
    err = EOSReadTopology(fac, topology);
    if (kIOErrNone != err) {
      PrintIOError(topology, err);
      goto bail;
    }
  }
bail:
  return err;
}

/********************************************************************************
 ********************************************************************************
 ********************************************************************************
 *****                              JSON Output                             *****
 ********************************************************************************
 ********************************************************************************
 ********************************************************************************/

/* The maximum number of elements to print in one row when printing an array */
#define MAX_ROW_SIZE 12

static void JSONPrintModes(uint32_t size, uint32_t numModes, const float *data, JSONWriter &wtr, const char *tag) {
  uint32_t modeSize = size / numModes;
  uint32_t maxCols = (modeSize < MAX_ROW_SIZE) ? modeSize : MAX_ROW_SIZE;
  wtr.openObject(tag);
  if (modeSize > 1) wtr.writeNumber(modeSize, "mode_size");
  if (numModes > 1) wtr.writeNumber(numModes, "num_modes");
  wtr.writeNumericArray(size, data, maxCols, "data");
  wtr.closeObject();
}

static void JSONPrintShapeModel(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  wtr.openObject(tag);
  JSONPrintModes(fac->getShapeMeanSize(), 1, fac->getShapeMean(), wtr, "mean");  // This is really a vector
  JSONPrintModes(fac->getShapeModesSize(), fac->getShapeNumModes(), fac->getShapeModes(), wtr, "pca_basis");
  JSONPrintModes(fac->getShapeEigenvaluesSize(), fac->getShapeEigenvaluesSize(), fac->getShapeEigenvalues(), wtr,
                 "eigenvalues");
  wtr.writeNumericArray(fac->getTriangleListSize(), fac->getTriangleList(), 3, "triangles");
  wtr.closeObject();
}

static void JSONPrintColorModel(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  wtr.openObject(tag);
  JSONPrintModes(fac->getColorMeanSize(), 1, fac->getColorMean(), wtr, "mean");  // This is really a vector
  JSONPrintModes(fac->getColorModesSize(), fac->getColorNumModes(), fac->getColorModes(), wtr, "pca_basis");
  JSONPrintModes(fac->getColorEigenvaluesSize(), fac->getColorEigenvaluesSize(), fac->getColorEigenvalues(), wtr,
                 "eigenvalues");
  wtr.writeNumericArray(fac->getTriangleListSize(), fac->getTriangleList(), 3, "triangles");
  wtr.closeObject();
}

static void JSONPrintMorphableModel(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  wtr.openObject(tag);
  JSONPrintShapeModel(fac, wtr, "shape_model");
  if (fac->getColorMeanSize() + fac->getColorModesSize() + fac->getColorEigenvaluesSize() != 0)
    JSONPrintColorModel(fac, wtr, "color_model");  // Don't print color unless there is something there
  if (fac->getTextureCoordinatesSize())
    wtr.writeNumericArray(fac->getTextureCoordinatesSize(), fac->getTextureCoordinates(), 2, "texture_coordinates");
  wtr.closeObject();
}

static void JSONPrintBlendShapes(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  uint32_t i, n;
  wtr.openArray(tag);
  for (i = 0, n = fac->getNumBlendShapes(); i < n; ++i) {
    wtr.openObject(nullptr);
    wtr.writeString(fac->getBlendShapeName(i), "name");
    JSONPrintModes(fac->getBlendShapeSize(i), 1, fac->getBlendShape(i), wtr, "blend_shape");  // This is really a vector
    wtr.closeObject();
  }
  wtr.closeArray();
}

static void JSONPrintIbugMappings(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  uint32_t n;
  wtr.openObject(tag);
  wtr.writeNumericArray(fac->getIbugLandmarkMappingsSize(), fac->getIbugLandmarkMappings(), 2, "landmark_mappings");
  n = (fac->getIbugRightContourSize());
  wtr.writeNumericArray(n, fac->getIbugRightContour(), n, "right_contour");  // straight across
  n = fac->getIbugLeftContourSize();
  wtr.writeNumericArray(n, fac->getIbugLeftContour(), n, "left_contour");  // straight across
  wtr.closeObject();
}

static FaceIOErr JSONPrintContours(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  wtr.openObject(tag);
  wtr.writeNumericArray(fac->getModelRightContourSize(), fac->getModelRightContour(),
                        fac->getModelRightContourSize(), "right_contour");
  wtr.writeNumericArray(fac->getModelLeftContourSize(),  fac->getModelLeftContour(),
                        fac->getModelLeftContourSize(), "left_contour");
  wtr.closeObject();
  return kIOErrNone;
}

static FaceIOErr JSONPrintTopology(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  if (!tag) tag = "edge_topology";
  wtr.openObject(tag);
  wtr.writeNumericArray(fac->getAdjacentFacesSize(), fac->getAdjacentFaces(), 2, "adjacent_faces");
  wtr.writeNumericArray(fac->getAdjacentVerticesSize(), fac->getAdjacentVertices(), 2, "adjacent_vertices");
  wtr.closeObject();
  return kIOErrNone;
}

static void JSONPrintNvlm(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  uint32_t n;
  wtr.openObject(tag);
  wtr.writeNumericArray(fac->getNvlmLandmarksSize(), fac->getNvlmLandmarks(), 2, "landmark_mapping");
  n = (fac->getNvlmRightContourSize());
  wtr.writeNumericArray(n, fac->getNvlmRightContour(), n, "right_contour"); // straight across
  n = fac->getNvlmLeftContourSize();
  wtr.writeNumericArray(n, fac->getNvlmLeftContour(), n, "left_contour");   // straight across
  wtr.closeObject();
}

static void JSONPrintPartitions(const FaceIOAdapter *fac, JSONWriter &wtr, const char *tag) {
  uint32_t i, n;
  wtr.openArray(tag);
  for (i = 0, n = fac->getNumPartitions(); i < n; ++i) {
    uint32_t partitionIndex, faceIndex, numFaces, vertexIndex, numVertices;
    int32_t smoothingGroup;
    const char *str;
    wtr.openObject(nullptr);
    partitionIndex = fac->getPartition(i, &faceIndex, &numFaces, &vertexIndex, &numVertices, &smoothingGroup);
    wtr.writeNumber(partitionIndex, "partition_index");
    wtr.writeNumber(faceIndex, "face_index");
    wtr.writeNumber(numFaces, "num_faces");
    wtr.writeNumber(vertexIndex, "vertex_index");
    wtr.writeNumber(numVertices, "num_vertices");
    wtr.writeNumber(smoothingGroup, "smoothing_group");
    if (nullptr != (str = fac->getPartitionName(i)) && str[0])
      wtr.writeString(str, "name");
    if (nullptr != (str = fac->getPartitionMaterialName(i)) && str[0])
      wtr.writeString(str, "material");
    wtr.closeObject();
  }
  wtr.closeArray();
}


/********************************************************************************
 * API                      PrintJSONFaceModel                              API *
 ********************************************************************************/

FaceIOErr PrintJSONFaceModel(FaceIOAdapter *fac, const char *file) {
  FaceIOErr err;
  JSONWriter wtr;

  err = wtr.open(file);
  if (kIOErrNone != err) {
    PrintIOError(file, err);
    return err;
  }

  wtr.openObject();
  JSONPrintMorphableModel(fac, wtr, "morphable_model");
  JSONPrintIbugMappings(fac, wtr, "ibug_mappings");
  JSONPrintBlendShapes(fac, wtr, "blend_shapes");
  JSONPrintContours(fac, wtr, "contours");
  JSONPrintTopology(fac, wtr, "edge_topology");
  if (fac->getNvlmLandmarksSize())
    JSONPrintNvlm(fac, wtr, "nvidia_mappings");
  if (fac->getNumPartitions())
    JSONPrintPartitions(fac, wtr, "partitions");
  wtr.closeObject();

  return err;
}
