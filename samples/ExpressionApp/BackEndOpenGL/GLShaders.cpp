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

#ifdef _MSC_VER
  #include "glad/glad.h"
#else
  #include <GLES3/gl3.h>
#endif // _MSC_VER
#include <string>
#include <stdint.h>
#include "GLShaders.h"


enum {
  myErrNone       = 0,
  myErrShader     = -1,
  myErrProgram    = -2,
  myErrTexture    = -3,
};


#define BAIL_IF_ERR(err)    do { if ((err)) { goto bail; } } while(0)
#define _STRINGIFY_(token)  #token
#define STRINGIFY(token)    _STRINGIFY_(token)
#define MAYBE_UNUSED(token) if (token){}


/****************************************************************************//**
 * Print Shader Log.
 *	\param[in]	id		the ID of the shader.
 *	\param[in]	type	either GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
 *	\param[in]	shader	the shader source code.
 ********************************************************************************/

static void PrintShaderLog(GLuint id, GLenum type, const char *shader) {
  GLsizei	msgLength;
  std::string errMsg;
  glGetShaderiv(id, GL_INFO_LOG_LENGTH, &msgLength);
  errMsg.resize(msgLength);
  glGetShaderInfoLog(id, msgLength, &msgLength, &errMsg[0]);
  fprintf(stderr, "\nShader Log:\n%sfor %s Shader:\n%s\n", errMsg.c_str(),
    ((type == GL_VERTEX_SHADER) ? "Vertex" : "Fragment"), shader);
}


/****************************************************************************//**
 * Print Program Log.
 *	\param[in]	id	the id of the program.
 ********************************************************************************/

static void PrintProgramLog(GLuint id) {
  GLsizei	msgLength;
  std::string errMsg;
  glGetProgramiv(id, GL_INFO_LOG_LENGTH, &msgLength);
  errMsg.resize(msgLength);
  glGetProgramInfoLog(id, msgLength, &msgLength, &errMsg[0]);
  fprintf(stderr, "\nProgram Log:\n%s\n", errMsg.c_str());
}


/****************************************************************************//**
 * NewShader
 ********************************************************************************/

static int NewShader(const char *shaderStr, GLenum type, GLuint *shaderID) {
  GLuint	id;
  GLint	result;

  *shaderID = 0;
  id = glCreateShader(type);
  glShaderSource(id, 1, &shaderStr, NULL);
  glCompileShader(id);
  glGetShaderiv(id, GL_COMPILE_STATUS, &result);
  if (result) {
    *shaderID = id;
    return myErrNone;
  }
  else {
    PrintShaderLog(id, type, shaderStr);
    glDeleteShader(id);
    return myErrShader;
  }
}


/****************************************************************************//**
 * NewProgram
 ********************************************************************************/

static int NewProgram(GLuint vertexShader, GLuint fragmentShader, GLuint *progID) {
  GLint   result;
  GLuint  id;

  *progID = 0;

  id = glCreateProgram();
  glAttachShader(id, vertexShader);
  glAttachShader(id, fragmentShader);

  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &result);
  if (result) {
    *progID = id;
    return myErrNone;
  }
  else {
    PrintProgramLog(id);
    glDeleteProgram(id);
    return myErrProgram;
  }
}


/****************************************************************************//**
 * IndexTypeFromSize
 ********************************************************************************/

static GLenum IndexTypeFromSize(unsigned indexSize) {
  return  (indexSize <  2) ? GL_UNSIGNED_BYTE
        : (indexSize == 2) ? GL_UNSIGNED_SHORT
        : GL_UNSIGNED_INT;
}


/********************************************************************************
 ********************************************************************************
 *****                       SMOOTH RENDERER                                *****
 ********************************************************************************
 ********************************************************************************/


/********************************************************************************
 * Shaders
 ********************************************************************************/

const char SmoothRenderer::_vertexShader[] =
  "uniform mat4 MVP;\n"
  "attribute vec3 vCol;\n"
  "attribute vec3 vPos;\n"
  "varying vec3 color;\n"
  "void main()\n"
  "{\n"
  "  gl_Position = MVP * vec4(vPos, 1.0);\n"
  "  color = vCol;\n"
  "}\n";
const char SmoothRenderer::_fragmentShader[] =
  "varying vec3 color;\n"
  "void main()\n"
  "{\n"
  "  gl_FragColor = vec4(color, 1.0);\n"
  "}\n";


/********************************************************************************
 * startup
 ********************************************************************************/

int SmoothRenderer::startup() {
  int err = myErrNone;
  GLuint vertexShader = 0, fragmentShader = 0;

  _programID = 0;
  BAIL_IF_ERR(err = NewShader(_vertexShader, GL_VERTEX_SHADER, &vertexShader));
  BAIL_IF_ERR(err = NewShader(_fragmentShader, GL_FRAGMENT_SHADER, &fragmentShader));
  BAIL_IF_ERR(err = NewProgram(vertexShader, fragmentShader, &_programID));
  _maxtrixID = glGetUniformLocation(_programID, "MVP");
  _vtxPosID = glGetAttribLocation(_programID, "vPos");
  _vtxColID = glGetAttribLocation(_programID, "vCol");
  err = (-1 == _maxtrixID || -1 == _vtxPosID || -1 == _vtxColID) ? myErrShader : myErrNone;

bail:
  if (myErrNone != err) shutdown();
  if (fragmentShader)   glDeleteShader(fragmentShader);
  if (vertexShader)     glDeleteShader(vertexShader);
  return err;
}


/********************************************************************************
 * use
 ********************************************************************************/

int SmoothRenderer::use() {
  if (0 == _programID)
    return myErrProgram;
  glUseProgram(_programID);
  return myErrNone;
}


/********************************************************************************
 * drawElements, from user memory
 ********************************************************************************/

void SmoothRenderer::drawElements(GLsizei numVertices, const GLfloat *positions, const GLfloat *colors, 
    GLenum graphicsMode, GLsizei indexCount, GLenum indexType, const GLvoid *indices, const GLfloat *M
) {
  MAYBE_UNUSED(numVertices);
  glUseProgram(_programID);
  if (M)
    glUniformMatrix4fv(_maxtrixID, 1, GL_FALSE, M);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  if (colors) {     /* Separate array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(*positions), positions);
    glEnableVertexAttribArray(_vtxColID);
    glVertexAttribPointer(_vtxColID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(*colors), colors);
  }
  else {            /* One contiguous array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(*positions), positions);
    glEnableVertexAttribArray(_vtxColID);
    glVertexAttribPointer(_vtxColID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(*positions), positions + 3);
  }
  glDrawElements(graphicsMode, indexCount, indexType, indices);
}


/********************************************************************************
 * drawElements, from buffer objects
 ********************************************************************************/

void SmoothRenderer::drawElements(GLuint vtxBuf, unsigned posOff, unsigned colOff,
    GLenum graphicsMode, GLsizei numIndices, unsigned indexSize, GLuint topoBuf, const GLfloat *M
) {
  GLenum  err;
  glUseProgram(_programID);
  if (M)
    glUniformMatrix4fv(_maxtrixID, 1, GL_FALSE, M);
  glBindBuffer(GL_ARRAY_BUFFER, vtxBuf);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, topoBuf);
  if (!(colOff == 12 || colOff == 0)) {   /* Separate array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(intptr_t)posOff);
    glEnableVertexAttribArray(_vtxColID);
    glVertexAttribPointer(_vtxColID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(intptr_t)colOff);
  }
  else {                                  /* One contiguous array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(intptr_t)(posOff + 0 * sizeof(float)));
    glEnableVertexAttribArray(_vtxColID);
    glVertexAttribPointer(_vtxColID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(intptr_t)(posOff + 3 * sizeof(float)));
    err = glGetError(); if (err) printf("glVertexAttribPointer returns %d\n", err);
  }
  glDrawElements(graphicsMode, numIndices, IndexTypeFromSize(indexSize), (void*)0);
}




/********************************************************************************
 ********************************************************************************
 *****                       TEXTURE RENDERER                               *****
 ********************************************************************************
 ********************************************************************************/



/********************************************************************************
 * shaders
 ********************************************************************************/

const char TextureRenderer::_vertexShader[] =
  "uniform mat4 MVP;\n"           // Model, view, projection matrices, concatenated.
  "attribute vec3 vPos;\n"        // Vertex position
  "attribute vec2 vTex;\n"        // Vertex texture coordinate
  "varying vec2 texCoord;\n"      // Interpolated texture coordinate
  "void main()\n"
  "{\n"
  "  gl_Position = MVP * vec4(vPos, 1.0);\n"
  "  texCoord = vTex;\n"
  "}\n";
const char TextureRenderer::_fragmentShader[] =
  "uniform sampler2D tex;\n"
  "varying vec2 texCoord;\n"      // Interpolated texture coordinate
  "void main()\n"
  "{\n"
  "  gl_FragColor = texture2D(tex, texCoord);\n"
  "}\n";


/********************************************************************************
 * startup
 ********************************************************************************/

int TextureRenderer::startup() {
  int err = myErrNone;
  GLuint vertexShader = 0, fragmentShader = 0;

  if (_programID)
    return myErrNone;
  _programID = 0;
  BAIL_IF_ERR(err = NewShader(_vertexShader, GL_VERTEX_SHADER, &vertexShader));
  BAIL_IF_ERR(err = NewShader(_fragmentShader, GL_FRAGMENT_SHADER, &fragmentShader));
  BAIL_IF_ERR(err = NewProgram(vertexShader, fragmentShader, &_programID));
  _maxtrixID = _vtxPosID = _vtxTexID = -1;
  _maxtrixID = glGetUniformLocation(_programID, "MVP");
  _vtxPosID = glGetAttribLocation(_programID, "vPos");
  _vtxTexID = glGetAttribLocation(_programID, "vTex");
  err = (-1 == _maxtrixID || -1 == _vtxPosID || -1 == _vtxTexID) ? myErrShader : myErrNone;

bail:
  if (myErrNone != err) shutdown();
  if (fragmentShader)   glDeleteShader(fragmentShader);
  if (vertexShader)     glDeleteShader(vertexShader);
  return err;
}


/********************************************************************************
 * use
 ********************************************************************************/

int TextureRenderer::use() {
  if (0 == _programID)
    return myErrProgram;
  glUseProgram(_programID);
  return myErrNone;
}


/********************************************************************************
 * drawElements, from user buffers
 ********************************************************************************/

void TextureRenderer::drawElements(GLsizei numVertices, const GLfloat *xyz, const GLfloat *uv,
    GLenum graphicsMode, GLsizei indexCount, GLenum indexType, const GLvoid *indices, GLuint texID ,const GLfloat *M
) {
  MAYBE_UNUSED(numVertices);
  glUseProgram(_programID);
  if (M)
    glUniformMatrix4fv(_maxtrixID, 1, GL_FALSE, M);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, texID);

  if (uv) {   /* Separate array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(*xyz), xyz);
    glEnableVertexAttribArray(_vtxTexID);
    glVertexAttribPointer(_vtxTexID, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(*uv), uv);
  }
  else {      /* One contiguous array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(*xyz), xyz);
    glEnableVertexAttribArray(_vtxTexID);
    glVertexAttribPointer(_vtxTexID, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(*xyz), xyz + 3);
  }
  glDrawElements(graphicsMode, indexCount, indexType, indices);
}


/********************************************************************************
 * drawElements, from buffer objects
 ********************************************************************************/

void TextureRenderer::drawElements(
    GLuint vtxBuf, unsigned xyzOff, unsigned uvOff,  GLenum graphicsMode,
    GLsizei numIndices, GLenum indexSize, GLuint indexBuf, GLuint texID, const GLfloat *M
) {
  GLenum  err;
  glUseProgram(_programID);
  if (M)
    glUniformMatrix4fv(_maxtrixID, 1, GL_FALSE, M);
  glBindBuffer(GL_ARRAY_BUFFER, vtxBuf);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
  glBindTexture(GL_TEXTURE_2D, texID);

  if (!(uvOff == 12 || uvOff == 0)) {   /* Separate array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(intptr_t)xyzOff);
    glEnableVertexAttribArray(_vtxTexID);
    glVertexAttribPointer(_vtxTexID, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)(intptr_t)uvOff);
  }
  else {                                /* One contiguous array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(intptr_t)(xyzOff + 0 * sizeof(float)));
    glEnableVertexAttribArray(_vtxTexID);
    glVertexAttribPointer(_vtxTexID, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(intptr_t)(xyzOff + 3 * sizeof(float)));
    err = glGetError(); if (err) printf("glVertexAttribPointer returns %d\n", err);
  }
  glDrawElements(graphicsMode, numIndices, IndexTypeFromSize(indexSize), (void*)0);
}


/********************************************************************************
 * drawQuad, from user buffers
 ********************************************************************************/

void TextureRenderer::drawQuad(const float xyz[4*3], const float uv[4*2], GLuint texID, const float *M) {
  static const unsigned char indices[4] = { 0, 1, 2, 3 }; // These need to be static, because GL is asynchronous
  drawElements(4, xyz, uv, GL_TRIANGLE_FAN, 4, GL_UNSIGNED_BYTE, indices, texID, M);
}


/********************************************************************************
 * Mesh shader
 ********************************************************************************/

class MeshShader {
  static const char _vertexShader[], _fragmentShader[];
};
// The ambient and diffuse coefficients are rolled into the ambCol and litCol, respectively.
const char MeshShader::_vertexShader[] =
  "uniform mat4 MVP;\n"           // Model, view, projection matrices, concatenated.
  "uniform mat3 N;\n"             // Normal matrix.
  "uniform vec3 litDir;\n"        // Light direction
  "uniform vec3 litCol;\n"        // Light color multiplied by the diffuse coefficient
  "uniform vec3 ambCol;\n"        // Ambient color multiplied by the ambient coefficient
  "attribute vec3 vtxPos;\n"      // Vertex position
  "attribute vec3 vtxNor;\n"      // Vertex normal
  "attribute vec2 vtxTex;\n"      // Vertex texture coordinate
  "varying vec2 texCoord;\n"      // Interpolated texture coordinate
  "varying vec3 illum;\n"         // Interpolated illumination
  "void main()\n"
  "{\n"
  "  gl_Position = MVP * vec4(vtxPos, 1.0);\n"
  "  texCoord = vtxTex;\n"
  "  illum = max(dot(N * vtxNor, litDir) * litCol + ambCol;\n"
  "}\n";



/********************************************************************************
 * UpdateTexture
 ********************************************************************************/

GLenum UpdateTexture(GLint texID, GLsizei width, GLsizei height, GLsizei rowBytes, GLenum glFormat, const GLvoid *pixels) {
  glBindTexture(GL_TEXTURE_2D, texID);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, rowBytes / 4);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, glFormat, GL_UNSIGNED_BYTE, pixels);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); // restore to default
  return glGetError();
}


/********************************************************************************
 ********************************************************************************
 *****                          LAMBERTIAN RENDERER                         *****
 ********************************************************************************
 ********************************************************************************/


/********************************************************************************
 * Shaders
 ********************************************************************************/

const char LambertianRenderer::_vertexShader[] =
  "#version 120\n"
  "uniform mat4 M;\n"
  "uniform mat4 VP;\n"
  "uniform vec4 lightLoc[" STRINGIFY(LAMBERTIAN_NUM_LIGHTS) "];\n"
  "uniform vec3 lightColor[" STRINGIFY(LAMBERTIAN_NUM_LIGHTS) "];\n"
  "uniform vec3 Ka;\n"
  "uniform vec3 Kd;\n"
  "attribute vec3 vPos;\n"
  "attribute vec3 vNrm;\n"
  "varying vec3 color;\n"
  "void main()\n"
  "{\n"
  "  vec4 loc = M * vec4(vPos, 1.);\n"                               // Transform points into world space ...
  "  gl_Position = VP * loc;\n"                                      // ... and screen space
  "  vec3 N = normalize(mat3(M) * vNrm);\n"                          // Transform normal into world space, assuming isotropic scaling
  "  color = Ka;\n"                                                  // Initialize color to ambient
  "  for (int i = 0; i < " STRINGIFY(LAMBERTIAN_NUM_LIGHTS) "; ++i)\n"
  "  {\n"
  "    vec3 L = normalize(lightLoc[i].xyz - lightLoc[i].w * loc.xyz);\n" // Compute vector to light: w must be either 1 or 0
  "    float d = dot(L, N);\n"                                      // Lambertian lighting
  "    if (d > 0.)\n"                                               // If the light hits the outside surface, ...
  "      color += d * lightColor[i] * Kd;\n"                        // ... accumulate color from the light source
  "   }\n"
  "}\n";
const char LambertianRenderer::_fragmentShader[] =
  "varying vec3 color;\n"
  "void main()\n"
  "{\n"
  "    gl_FragColor = vec4(color, 1.);\n"                           // Interpolate the color
  "}\n";


/********************************************************************************
 * startup
 ********************************************************************************/

int LambertianRenderer::startup() {
  int err = myErrNone;
  GLuint vertexShader = 0, fragmentShader = 0;

  _programID = 0;
  BAIL_IF_ERR(err = NewShader(_vertexShader, GL_VERTEX_SHADER, &vertexShader));
  BAIL_IF_ERR(err = NewShader(_fragmentShader, GL_FRAGMENT_SHADER, &fragmentShader));
  BAIL_IF_ERR(err = NewProgram(vertexShader, fragmentShader, &_programID));
  _lightLoc       = glGetUniformLocation(_programID, "lightLoc");     // light locations
  _lightColor     = glGetUniformLocation(_programID, "lightColor");   // light diffuse colors
  _MmatrixID      = glGetUniformLocation(_programID, "M");            // M matrix; require UL 3x3 to be orthogonal
  _VPmatrixID     = glGetUniformLocation(_programID, "VP");           // VP matrix
  _ambientColorID = glGetUniformLocation(_programID, "Ka");           // ambient color
  _diffuseColorID = glGetUniformLocation(_programID, "Kd");           // diffuse color
  _vtxPosID       = glGetAttribLocation(_programID, "vPos");          // vertex positions
  _vtxNrmID       = glGetAttribLocation(_programID, "vNrm");          // vertex normals

  err = (-1 == _lightLoc || -1 == _lightColor || -1 == _MmatrixID || -1 == _VPmatrixID || -1 == _ambientColorID
      || -1 == _diffuseColorID || -1 == _vtxPosID || -1 == _vtxNrmID) ? myErrShader : myErrNone;
  BAIL_IF_ERR(err);

bail:
  if (myErrNone != err) shutdown();
  if (fragmentShader)   glDeleteShader(fragmentShader);
  if (vertexShader)     glDeleteShader(vertexShader);
  return err;
}


/********************************************************************************
 * use
 ********************************************************************************/

int LambertianRenderer::use() {
  if (0 == _programID)
    return myErrProgram;
  glUseProgram(_programID);
  return myErrNone;
}


/********************************************************************************
 * set lights
 ********************************************************************************/

void LambertianRenderer::setLights(const float locXYZW[4*LAMBERTIAN_NUM_LIGHTS], const float colorRGB[3*LAMBERTIAN_NUM_LIGHTS]) {
  glUseProgram(_programID);
  glUniform4fv(_lightLoc, LAMBERTIAN_NUM_LIGHTS, locXYZW);
  glUniform3fv(_lightColor, LAMBERTIAN_NUM_LIGHTS, colorRGB);
}



/********************************************************************************
 * drawElements, from user memory
 ********************************************************************************/

void LambertianRenderer::drawElements(GLsizei numVertices, const GLfloat *positions, const GLfloat *normals,
    GLenum graphicsMode, GLsizei indexCount, GLenum indexType, const GLvoid *indices,
    const GLfloat M[4*4], const GLfloat VP[4*4], const float Ka[3], const float Kd[3]
) {
  MAYBE_UNUSED(numVertices);
  glUseProgram(_programID);
  if (M)  glUniformMatrix4fv(_MmatrixID, 1, GL_FALSE, M);
  if (VP) glUniformMatrix4fv(_VPmatrixID, 1, GL_FALSE, VP);
  if (Ka) glUniform3fv(_ambientColorID, 1, Ka);
  if (Kd) glUniform3fv(_diffuseColorID, 1, Kd);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  if (normals) {  /* Separate array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(*positions), positions);
    glEnableVertexAttribArray(_vtxNrmID);
    glVertexAttribPointer(_vtxNrmID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(*normals), normals);
  }
  else {          /* One contiguous array for position and color */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(*positions), positions);
    glEnableVertexAttribArray(_vtxNrmID);
    glVertexAttribPointer(_vtxNrmID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(*normals), normals + 3);
  }
  glDrawElements(graphicsMode, indexCount, indexType, indices);
}


/********************************************************************************
 * drawElements, from buffer objects
 ********************************************************************************/

void LambertianRenderer::drawElements(GLuint vtxBuf, unsigned posOff, unsigned nrmOff,
    GLenum graphicsMode, GLsizei numIndices, unsigned indexSize, GLuint indexBuf,
    const GLfloat M[4*4], const GLfloat VP[4*4], const float Ka[3], const float Kd[3]
) {
  GLenum  err;

  glUseProgram(_programID);
  if (M)  glUniformMatrix4fv(_MmatrixID, 1, GL_FALSE, M);
  if (VP) glUniformMatrix4fv(_VPmatrixID, 1, GL_FALSE, VP);
  if (Ka) glUniform3fv(_ambientColorID, 1, Ka);
  if (Kd) glUniform3fv(_diffuseColorID, 1, Kd);

  glBindBuffer(GL_ARRAY_BUFFER, vtxBuf);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuf);
  if (!(nrmOff == 12 || nrmOff == 0)) {   /* Separate array for position and normal */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(intptr_t)posOff);
    glEnableVertexAttribArray(_vtxNrmID);
    glVertexAttribPointer(_vtxNrmID, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(intptr_t)nrmOff);
  }
  else {                                  /* One contiguous array for position and normal */
    glEnableVertexAttribArray(_vtxPosID);
    glVertexAttribPointer(_vtxPosID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(intptr_t)(posOff + 0 * sizeof(float)));
    glEnableVertexAttribArray(_vtxNrmID);
    glVertexAttribPointer(_vtxNrmID, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(intptr_t)(posOff + 3 * sizeof(float)));
    err = glGetError(); if (err) printf("glVertexAttribPointer returns %d\n", err);
  }
  glDrawElements(graphicsMode, numIndices, IndexTypeFromSize(indexSize), (void*)0);
}
