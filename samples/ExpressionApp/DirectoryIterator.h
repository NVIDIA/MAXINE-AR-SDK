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

#ifndef __DIRECTORY_ITERATOR_H
#define __DIRECTORY_ITERATOR_H

class DirectoryIterator {
public:
  enum {
    kTypeFile      = 1,
    kTypeDirectory = 2,
    kTypeSpecial   = 4,
    kTypeAll       = (kTypeFile | kTypeDirectory | kTypeSpecial)
  };

  /// Constructor
  DirectoryIterator();

  /// Constructor
  /// @param[in] path         The path of the directory to iterate.
  /// @param[in] iterateWhat  The types of files to list.
  DirectoryIterator(const char *path, unsigned iterateWhat);

  /// Destructor
  ~DirectoryIterator();

  /// Start looking in a particular directory.
  /// @param[in] path         The path of the directory to iterate.
  /// @param[in] iterateWhat  The types of files to list.
  /// @return   0   If successful,
  ///          -1   If path was NULL,
  ///         -99   If there are no files.
  int init(const char *path, unsigned iterateWhat);

  /// Get the next file.
  /// @param pName[out]   a place to store the name of the next file.
  /// @param type[out]    a place to store the type of the next file.
  /// @return   0   If successful,
  ///          -1   If path was NULL,
  ///         -99   If there are no more files.
  int next(const char **pName, unsigned *type);

private:
  struct Impl;
  Impl *m_impl;
};

#endif // __DIRECTORY_ITERATOR_H
