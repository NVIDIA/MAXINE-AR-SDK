/*###############################################################################
#
# Copyright 2016-2021 NVIDIA Corporation
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

#ifndef __ARSHADERS_H__
#define __ARSHADERS_H__

#ifdef _MSC_VER
  #include "glad/glad.h"
#else
  #include <GLES3/gl3.h>
#endif // _MSC_VER


/********************************************************************************
 ********************************************************************************
 *****                       SMOOTH RENDERER                                *****
 ********************************************************************************
 ********************************************************************************/

class SmoothRenderer {
public:

  SmoothRenderer() { _programID = 0; }
  ~SmoothRenderer() { shutdown(); }

  int     startup();
  void    shutdown() { if (_programID) glDeleteProgram(_programID); _programID = 0; }
  int     use();
  int     activate() { return startup(); }   // DEPRECATED
  void    deactivate() { shutdown(); }         // DEPRECATED

  /** These take vertex and topology data in user-space buffers.
   * @param[in]   numPts      The number of points in xyz or rgb.
   * @param[in]   xyz         The vertex locations {x, y, z }.
   * @param[in]   rgb         The vertex colors { r, g, b }, in [0, 1].
   * @param[in]   numIndices  The number of indices.
   * @param[in]   indices     The indices. Note that three versions are given, where indices can be 1, 2, or 4 bytes.
   * @param[in]   M           the matrix.
   */
  void drawTriMesh(unsigned numPts, const float* xyz, const float* rgb,
        unsigned numIndices, const unsigned char* indices, const float* M = nullptr) {
    drawElements(numPts, xyz, rgb, GL_TRIANGLES, numIndices, GL_UNSIGNED_BYTE, indices, M);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* rgb,
        unsigned numIndices, const unsigned short* indices, const float* M = nullptr) {
    drawElements(numPts, xyz, rgb, GL_TRIANGLES, numIndices, GL_UNSIGNED_SHORT, indices, M);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* rgb,
        unsigned numIndices, const unsigned int* indices, const float* M = nullptr) {
    drawElements(numPts, xyz, rgb, GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, indices, M);
  }

  /** These take vertex and topology data in GL buffer objects
   * @param[in]   vtxBuf      the vertex buffer object identifier.
   * @param[in]   xyzOff      the offset, in bytes, of the xyz positions in the vertex buffer.
   * @param[in]   rgbOff      the offset, in bytes, of the rgb   color   in the vertex buffer.
   * @param[in]   numIndices  the number of indices.
   * @param[in]   indexBuf    the index buffer object identifier.
   * @param[in]   indexSize   the byte size of the indices: 1, 2, or 4.
   * @param[in]   M           the matrix.
   */
  void drawTriMesh(GLuint vtxBuf, unsigned xyzOff, unsigned rgbOff,
        unsigned numIndices, GLuint indexBuf, GLenum indexSize, const float* M) {
    drawElements(vtxBuf, xyzOff, rgbOff, GL_TRIANGLES, numIndices, indexSize, indexBuf, M);
  }

private:
  /** Render geometry from user buffers, pre-shaded at vertices.
   * @param[in]   numVertices     The number of 3D vertices.
   * @param[in]   positions       The array of 3D positions -- one for every vertex.
   * @param[in]   colors          The array of RGB colors -- one for every vertex.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   indexCount      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexType       The type of index { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT }.
   * @param[in]   indices         The array of vertices.
   * @param[in]   M               The modeling-viewing-projection matrix.
   */
  void drawElements(GLsizei numVertices, const GLfloat* positions, const GLfloat* colors,
        GLenum graphicsMode, GLsizei indexCount, GLenum indexType, const GLvoid* indices, const GLfloat* M);

  /** Render geometry from GL buffer objects, pre-shaded at vertices.
   * @param[in]   vtxBuf          The ID of the GL buffer used to store the vertices.
   * @param[in]   posOff          The offset of the positions in the vertex buffer. This is typically 0,
   *                              but is not restricted so.
   * @param[in]   colOff          The offset of the colors in the vertex buffer. Both planar (homogeneous, separate)
   *                              and chunky (nonhomogeneous, interleaved) representations are accommodated.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   indexCount      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexType       The type of index { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT }.
   * @param[in]   indexBuf        The ID of the GL buffer used to store the indices.
   * @param[in]   M               The modeling-viewing-projection matrix.
   */
  void drawElements(GLuint vtxBuf, unsigned posOff, unsigned colOff,
        GLenum graphicsMode, GLsizei indexCount, GLenum indexType, GLuint indexBuf, const GLfloat* M);

  GLuint              _programID;
  GLint               _maxtrixID, _vtxPosID, _vtxColID;
  static const char   _vertexShader[], _fragmentShader[];
};


/********************************************************************************
 ********************************************************************************
 *****                       TEXTURE RENDERER                               *****
 ********************************************************************************
 ********************************************************************************/

class TextureRenderer {
public:

  TextureRenderer() { _programID = 0; }
  ~TextureRenderer() { shutdown(); }

  int   startup();
  void  shutdown() { if (_programID) glDeleteProgram(_programID); _programID = 0; }
  int   use();
  int   activate() { return startup(); }   // DEPRECATED
  void  deactivate() { shutdown(); }         // DEPRECATED

  /** These take vertex and topology data in user-space buffers.
   * @param[in]   numPts      The number of points in xyz or uv.
   * @param[in]   xyz         The vertex locations {x, y, z }.
   * @param[in]   uv          The vertex texture coordinates { u, v }, in [0, 1].
   * @param[in]   numIndices  The number of indices.
   * @param[in]   indices     The indices. Note that three versions are given, where indices can be 1, 2, or 4 bytes.
   * @param[in]   texID       The texture ID.
   * @param[in]   M           the matrix. NULL keeps the matrix as it was in the last invocation.
   */
  void drawTriMesh(unsigned numPts, const float* xyz, const float* uv,
        unsigned numIndices, const unsigned char* indices, GLuint texID, const float* M = nullptr) {
    drawElements(numPts, xyz, uv, GL_TRIANGLES, numIndices, GL_UNSIGNED_BYTE, indices, texID, M);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* uv,
        unsigned numIndices, const unsigned short* indices, GLuint texID, const float* M = nullptr) {
    drawElements(numPts, xyz, uv, GL_TRIANGLES, numIndices, GL_UNSIGNED_SHORT, indices, texID, M);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* uv,
        unsigned numIndices, const unsigned int* indices, GLuint texID, const float* M = nullptr) {
    drawElements(numPts, xyz, uv, GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, indices, texID, M);
  }

  /** Draw a texture-mapped quadrilateral.
   * @param[in]   xyz         The vertex locations {x, y, z }.
   * @param[in]   uv          The vertex texture coordinates { u, v }, in [0, 1].
   * @param[in]   texID       The texture ID.
   * @param[in]   M           the matrix. NULL keeps the matrix as it was in the last invocation.
   */
  void drawQuad(const float xyz[4 * 3], const float uv[4 * 2], GLuint texID, const float* M = nullptr);

  /** These take vertex and topology data in GL buffer objects
   * @param[in]   vtxBuf      the vertex buffer object identifier.
   * @param[in]   xyzOff      the offset, in bytes, of the     xyz positions   in the vertex buffer.
   * @param[in]   uvOff       the offset, in bytes, of the texture coordinates in the vertex buffer.
   * @param[in]   numIndices  the number of indices.
   * @param[in]   indexBuf    the index buffer object identifier.
   * @param[in]   indexSize   the byte size of the indices: 1, 2, or 4.
   * @param[in]   texID       The texture ID.
   * @param[in]   M           the matrix. NULL keeps the matrix as it was in the last invocation.
   */
  void drawTriMesh(GLuint vtxBuf, unsigned xyzOff, unsigned rgbOff,
        unsigned numIndices, GLuint indexBuf, GLenum indexSize, GLuint texID, const float* M) {
    drawElements(vtxBuf, xyzOff, rgbOff, GL_TRIANGLES, numIndices, indexSize, indexBuf, texID, M);
  }

private:
  /** Render geometry from user buffers.
   * @param[in]   numVertices     The number of 3D vertices.
   * @param[in]   positions       The array of 3D positions -- one for every vertex.
   * @param[in]   uv              The array of texture coordinates -- one for every vertex.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   indexCount      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexType       The type of index { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT }.
   * @param[in]   indices         The array of vertices.
   * @param[in]   texID           The ID of the texture to be used.
   * @param[in]   M               The modeling-viewing-projection matrix.
  */
  void drawElements(GLsizei numVertices, const GLfloat* positions, const GLfloat* uv, GLenum graphicsMode,
          GLsizei indexCount, GLenum indexType, const GLvoid* indices, GLuint texID, const GLfloat* M);

  /** Render geometry from GL buffer objects.
   * @param[in]   vtxBuf          The ID of the GL buffer used to store the vertices.
   * @param[in]   posOff          The offset of the positions in the vertex buffer. This is typically 0,
   *                              but is not restricted so.
   * @param[in]   uvOff           The offset of the texture coordinates in the vertex buffer. Both planar
   *                              (homogeneous, separate) and chunky (nonhomogeneous, interleaved)
   *                              representations are accommodated.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   indexCount      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexSize       The size of index { 1, 2, 4 } in bytes.
   * @param[in]   indexBuf        The ID of the GL buffer used to store the indices.
   * @param[in]   texID           The ID of the texture to be used.
   * @param[in]   M               The modeling-viewing-projection matrix.
   */
  void drawElements(GLuint vtxBuf, unsigned posOff, unsigned uvOff, GLenum graphicsMode,
        GLsizei indexCount, GLenum indexSize, GLuint indexBuf, GLuint texID, const GLfloat* M);

  GLuint              _programID;
  GLint               _maxtrixID, _vtxPosID, _vtxTexID;
  static const char   _vertexShader[], _fragmentShader[];
};


/** Update the specified texture.
 * @param[in]   texID       The ID of the texture to be updated.
 * @param[in]   width       The width  of the source image.
 * @param[in]   height      The height of the source image.
 * @param[in]   rowBytes    The byte stride between pixels vertically in the source image (must be positive).
 * @param[in]   glFormat    The format of the source image. One of { GL_RGBA, GL_BGRA, GL_RGB, GL_BGR, GL_RG, GL_R }.
 * @param[in]   pixels      A pointer to pixel(0,0) of the source image.
 * @return      GL_NO_ERROR if the update was successful.
 */
GLenum UpdateTexture(GLint texID, GLsizei width, GLsizei height, GLsizei rowBytes, GLenum glFormat,
        const GLvoid* pixels);


/********************************************************************************
 ********************************************************************************
 *****                         LAMBERTIAN SHADER                            *****
 ********************************************************************************
 ********************************************************************************/

 /** The camera is assumed to be at the origin. Lights are represented in the camera coordinate system.
  * The model coordinates are transformed M * pt;
  * the normal transformed by (M{3x3})^(-1)^(T), or simply M{3x3} assuming the scaling is isotropic.
  * We further assume that M{3x3} is orthonormal.
  */
class LambertianRenderer {
public:
#define LAMBERTIAN_NUM_LIGHTS 2

  LambertianRenderer() { _programID = 0; }
  ~LambertianRenderer() { shutdown(); }

  int     startup();
  void    shutdown() { if (_programID) glDeleteProgram(_programID); _programID = 0; }
  int     use();

  /** Set all lights. We accommodate point lights or directional lights.
   * These are specified in camera space, which we assume is fixed while the objects move.
   * @param[in]   locXYZW     The location of the lights -- in camera space.
   *                          The homogeneous coordinate W is used to choose between
   *                          directional lights (W=0) and point lights (W=1).
   *                          The result is undefined for other values of W.
   * @param[in]   colorRGB    the emissive color of the light source, RGB in [0, 1].
   *                          To turn a light off, set its emissive color to (0,0,0).
   */
  void    setLights(const float locXYZW[4 * LAMBERTIAN_NUM_LIGHTS], const float colorRGB[3 * LAMBERTIAN_NUM_LIGHTS]);

  /* These take vertex and topology data in user-space buffers.
   * @param[in]   numPts      The number of points in xyz or normals.
   * @param[in]   xyz         The vertex locations {x, y, z}.
   * @param[in]   nrm         The vertex normals {nx, ny, nz}.
   * @param[in]   numIndices  The number of indices.
   * @param[in]   indices     The indices. Note that three versions are given, where indices can be 1, 2, or 4 bytes.
   * @param[in]   M           The      modeling      matrix. If NULL, the previous matrix will be used.
   * @param[in]   VP          The viewing+projection matrix. If NULL, the previous matrix will be used.
   * @param[in]   Ka          The ambient color {r, g, b}. If NULL, the previous ambient color will be used.
   * @param[in]   Kd          The diffuse color {r, g, b}. If NULL, the previous diffuse color will be used.
   */
  void drawTriMesh(unsigned numPts, const float* xyz, const float* nrm, unsigned numIndices, const unsigned char* indices,
        const float* M = nullptr, const float* VP = nullptr, const float* Ka = nullptr, const float* Kd = nullptr) {
    drawElements(numPts, xyz, nrm, GL_TRIANGLES, numIndices, GL_UNSIGNED_BYTE, indices, M, VP, Ka, Kd);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* nrm, unsigned numIndices, const unsigned short* indices,
        const float* M = nullptr, const float* VP = nullptr, const float* Ka = nullptr, const float* Kd = nullptr) {
    drawElements(numPts, xyz, nrm, GL_TRIANGLES, numIndices, GL_UNSIGNED_SHORT, indices, M, VP, Ka, Kd);
  }
  void drawTriMesh(unsigned numPts, const float* xyz, const float* nrm, unsigned numIndices, const unsigned int* indices,
        const float* M = nullptr, const float* VP = nullptr, const float* Ka = nullptr, const float* Kd = nullptr) {
    drawElements(numPts, xyz, nrm, GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, indices, M, VP, Ka, Kd);
  }

  /* These take vertex and topology data in GL buffer objects
   * @param[in]   vtxBuf      the vertex buffer object identifier.
   * @param[in]   xyzOff      the offset, in bytes, of the xyz positions in the vertex buffer.
   * @param[in]   nrmOff      the offset, in bytes, of the normals       in the vertex buffer.
   * @param[in]   numIndices  the number of indices.
   * @param[in]   indexBuf    the index buffer object identifier.
   * @param[in]   indexSize   the byte size of the indices: 1, 2, or 4.
   * @param[in]   M           The      modeling      matrix. If NULL, the previous matrix will be used.
   * @param[in]   VP          The viewing+projection matrix. If NULL, the previous matrix will be used.
   * @param[in]   Ka          The ambient color {r, g, b}. If NULL, the previous ambient color will be used.
   * @param[in]   Kd          The diffuse color {r, g, b}. If NULL, the previous diffuse color will be used.
   */
  void drawTriMesh(GLuint vtxBuf, unsigned xyzOff, unsigned rgbOff, unsigned numIndices, GLuint indexBuf, GLenum indexSize,
        const float* M = nullptr, const float* VP = nullptr, const float* Ka = nullptr, const float* Kd = nullptr) {
    drawElements(vtxBuf, xyzOff, rgbOff, GL_TRIANGLES, numIndices, indexSize, indexBuf, M, VP, Ka, Kd);
  }

private:
  /** Render geometry from user buffers.
   * @param[in]   numVertices     The number of 3D vertices.
   * @param[in]   positions       The array of 3D positions -- one for every vertex.
   * @param[in]   normals         The array of normals -- one for every vertex.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   indexCount      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexType       The type of index { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_UNSIGNED_INT }.
   * @param[in]   indices         The array of vertices.
   * @param[in]   MV              The modeling-viewing matrix.
   * @param[in]   P               The projection matrix.
   * @param[in]   Ka              The ambient color.
   * @param[in]   Kd              The diffuse color.
   */
  void drawElements(GLsizei numVertices, const GLfloat* positions, const GLfloat* normals,
        GLenum graphicsMode, GLsizei indexCount, GLenum indexType, const GLvoid* indices,
        const GLfloat M[4 * 4], const GLfloat VP[4 * 4], const float Ka[3], const float Kd[3]);

  /** Render geometry from GL buffer objects.
   * @param[in]   vtxBuf          The ID of the GL buffer used to store the vertices.
   * @param[in]   posOff          The offset of the positions in the vertex buffer. This is typically 0,
   *                              but is not restricted so.
   * @param[in]   nrmOff          The offset of the normals in the vertex buffer. Both planar (homogeneous, separate)
   *                              and chunky (nonhomogeneous, interleaved) representations are accommodated.
   * @param[in]   graphicsMode    One of { GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN }.
   *                              GL_QUADS is not supported.
   * @param[in]   numIndices      The number of 0-based vertex indices that define the geometry
   *                              from the vertices and graphics mode.
   * @param[in]   indexSize       The size of index { 1, 2, 4 } in bytes.
   * @param[in]   indexBuf        The ID of the GL buffer used to store the indices.
   * @param[in]   M               The modeling matrix.
   * @param[in]   VP              The viewing+projection matrix.
   * @param[in]   Ka              The ambient color.
   * @param[in]   Kd              The diffuse color.
   */
  void drawElements(GLuint vtxBuf, unsigned posOff, unsigned nrmOff,
        GLenum graphicsMode, GLsizei numIndices, unsigned indexSize, GLuint indexBuf,
        const GLfloat M[4 * 4], const GLfloat VP[4 * 4], const float Ka[3], const float Kd[3]);

  GLuint              _programID;
  GLint               _MmatrixID, _VPmatrixID, _lightLoc, _lightColor, _ambientColorID, _diffuseColorID;
  GLint               _vtxPosID, _vtxNrmID;
  static const char   _vertexShader[], _fragmentShader[];
};


#endif /* __ARSHADERS_H__ */
