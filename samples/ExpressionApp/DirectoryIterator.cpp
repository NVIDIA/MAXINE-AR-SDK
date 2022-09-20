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

#include "DirectoryIterator.h"


#ifdef _WIN32
////////////////////////////////////////////////////////////////////////////////
/////                               WINDOWS                                /////
////////////////////////////////////////////////////////////////////////////////
#include <Windows.h>
#include <string>

struct DirectoryIterator::Impl {
  HANDLE            h;
  unsigned          which;
  bool              first;
  WIN32_FIND_DATAA  data;
};

DirectoryIterator::DirectoryIterator() {
  m_impl = new DirectoryIterator::Impl;
  m_impl->h = nullptr;
}

DirectoryIterator::DirectoryIterator(const char *path, unsigned iterateWhat) : DirectoryIterator() {
  init(path, iterateWhat);
}

DirectoryIterator::~DirectoryIterator() {
  if (m_impl) {
    if (m_impl->h) FindClose(m_impl->h);
    delete m_impl;
  }
}

int DirectoryIterator::init(const char *path, unsigned iterateWhat) {
  std::string pathStar = path;
  pathStar += "\\*";
  if (nullptr == (m_impl->h = FindFirstFileA(pathStar.c_str(), &m_impl->data))) return -99; /* either dir or file */
  m_impl->which = iterateWhat ? iterateWhat : kTypeAll;
  m_impl->first = true;
  return 0;
}

int DirectoryIterator::next(const char **pName, unsigned *type) {
  if (!pName) return -1;
  while (1) {
    if (m_impl->first) {
      m_impl->first = false;
    }
    else if (!FindNextFileA(m_impl->h, &m_impl->data)) {
      *pName = nullptr;
      if (type) *type = 0;
      return -99;
    }
    *pName = m_impl->data.cFileName;

    if (0 != (m_impl->data.dwFileAttributes & (
      FILE_ATTRIBUTE_NORMAL              |
      FILE_ATTRIBUTE_ARCHIVE             |
      FILE_ATTRIBUTE_COMPRESSED          |
      FILE_ATTRIBUTE_ENCRYPTED           |
      FILE_ATTRIBUTE_HIDDEN              |
      FILE_ATTRIBUTE_INTEGRITY_STREAM    |
      FILE_ATTRIBUTE_NOT_CONTENT_INDEXED |
      FILE_ATTRIBUTE_NO_SCRUB_DATA       |
      FILE_ATTRIBUTE_READONLY            |
      FILE_ATTRIBUTE_REPARSE_POINT       |
      FILE_ATTRIBUTE_SPARSE_FILE         |
      FILE_ATTRIBUTE_TEMPORARY
    ))) {
      if (m_impl->which & kTypeFile) {
        if (type) *type = kTypeFile;
        break;
      }
    }

    else if (0 == (m_impl->data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
      if (m_impl->which & kTypeDirectory) {
        if (type) *type = kTypeDirectory;
        break;
      }
    }

    else {
      if (m_impl->which & kTypeSpecial) {
        if (type) *type = kTypeSpecial;
        break;
      }
    }
  }

  return 0;
}


#else /* !_WIN32 == UNIX */
////////////////////////////////////////////////////////////////////////////////
/////                                 UNIX                                 /////
////////////////////////////////////////////////////////////////////////////////
#include <dirent.h>

struct DirectoryIterator::Impl {
  DIR      *dp;
  unsigned which;
};

DirectoryIterator::DirectoryIterator() {
  m_impl = new DirectoryIterator::Impl;
  m_impl->dp = nullptr;
}

DirectoryIterator::DirectoryIterator(const char* path, unsigned iterateWhat) : DirectoryIterator() {
  init(path, iterateWhat);
}

DirectoryIterator::~DirectoryIterator() {
  if (m_impl) {
    if (m_impl->dp) closedir(m_impl->dp);
    delete m_impl;
  }
}

int DirectoryIterator::init(const char *path, unsigned iterateWhat)  {
  if (nullptr == (m_impl->dp = opendir(path))) return -1;
  m_impl->which = iterateWhat ? iterateWhat : kTypeAll;
  return 0;
}

int DirectoryIterator::next(const char **pName, unsigned *type)  {
  struct dirent *entry;

  if (type) *type = 0;
  if (!pName) return -1;
  while (nullptr != (entry = readdir(m_impl->dp))) {
    *pName = entry->d_name;
    switch (entry->d_type) {
      case DT_REG:  if (m_impl->which & kTypeFile)      { if (type) *type = kTypeFile;       return 0; } break;
      case DT_DIR:  if (m_impl->which & kTypeDirectory) { if (type) *type = kTypeDirectory;  return 0; } break;
      default:      if (m_impl->which & kTypeSpecial)   { if (type) *type = kTypeSpecial;    return 0; } break;
    }
  }
  *pName = nullptr;
  return -99;
}


#endif /* UNIX */
