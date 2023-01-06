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

#include "OpenGLMeshRenderer.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include "glad/glad.h"
#define strcasecmp _stricmp
#else
#include <GLES3/gl3.h>
#endif  // _MSC_VER

#include "FaceIO.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/quaternion.hpp"
#include <glm/gtc/type_ptr.hpp>
#include "GLMaterial.h"
#include "GLMesh.h"
#include "GLShaders.h"
#include "GLSpectrum.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/highgui/highgui.hpp"
#include "SimpleFaceModel.h"


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                    SUPPORT MACROS AND FUNCTIONS                      /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define BAIL_IF_ERR(err)  do { if ((err) != 0)  { goto bail; } } while(0)
#define BAIL(err, code)   do {     err = code;    goto bail;   } while(0)

#ifndef __BYTE_ORDER__                  /* How bytes are packed into a 32 bit word */
  #define __ORDER_LITTLE_ENDIAN__ 3210  /* First byte in the least significant position */
  #define __ORDER_BIG_ENDIAN__    0123  /* First byte in the most  significant position */
#if defined(__amd64__) || defined(__amd64) || defined(__x86_64__) || defined(__x86_64) || defined(_M_AMD64) || _MSC_VER
  #define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
  #endif /* _MSC_VER */
#endif /* __BYTE_ORDER__ */


/********************************************************************************
 * glfwErrorCallback
 ********************************************************************************/

static void glfwErrorCallback(int error, const char *description) {
  fprintf(stderr, "Error %d: %s\n", error, description);
}


/********************************************************************************
 * MakeGLContext
 ********************************************************************************/

static NvCV_Status MakeGLContext(int width, int height, const char *title, GLFWwindow **pWindow) {
  NvCV_Status nvErr = NVCV_SUCCESS;
  GLFWwindow *window;

  /* Get a context */
  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit()) {
    // Initialization failed
    fprintf(stderr, "Unable to initialize glfw\n");
    return NVCV_ERR_INITIALIZATION;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  window = glfwCreateWindow(width, height, title, /*GLFWmonitor */NULL, /*GLFWwindow*/NULL);
  if (!window) {
    // Window or OpenGL context creation failed
    fprintf(stderr, "Unable to create glfw window\n");
    BAIL(nvErr, NVCV_ERR_INITIALIZATION);
  }
  int winWidth, winHeight;
  glfwGetWindowSize(window, &winWidth, &winHeight);
  if (winWidth != width || winHeight != height) {
    fprintf(stderr, "getWindowSize(%u x %u) != (%u x %u)\n", winWidth, winHeight, width, height);
  }
  glfwMakeContextCurrent(window);
  #ifdef _MSC_VER
    if (!gladLoadGL()) {
      fprintf(stderr, "Unable to load GL\n");
      BAIL(nvErr, NVCV_ERR_INITIALIZATION);
    }
    fprintf(stderr, "OpenGL Version %d.%d loaded\n", GLVersion.major, GLVersion.minor);
  #endif // _MSC_VER
  *pWindow = window;

bail:
  return nvErr;
}


/********************************************************************************
 * CloseGLContext
 ********************************************************************************/

static void CloseGLContext(GLFWwindow *window) {
  if (window)
    glfwDestroyWindow(window);
  glfwTerminate();
}


/********************************************************************************
 * ComputeDualTopologyFromAdjacencies
 ********************************************************************************/

static NvCV_Status ComputeDualTopologyFromAdjacencies(const SimpleFaceModelAdapter *fma, GLMesh *mesh) {
  union IVF {
    unsigned i;
    struct VF {
      #if      __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        unsigned short face, vertex;                  // Vertex in most significant position
      #else // __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        unsigned short vertex, face;                  // Vertex in most significant position
      #endif // __BYTE_ORDER__
    } vf;
    bool operator<( const IVF& other) { return i <  other.i; }
    bool operator==(const IVF& other) { return i == other.i; }
  };
  std::vector<IVF> topo;

  topo.reserve(mesh->numVertices() * 6 * 2);          // Assume valence-6, duplicated
  const unsigned short *adjVertices = const_cast<SimpleFaceModelAdapter*>(fma)->getAdjacentVertices(0),
                       *adjFaces    = const_cast<SimpleFaceModelAdapter*>(fma)->getAdjacentFaces(0);
  unsigned n = fma->getAdjacentVerticesSize();
  if (n != fma->getAdjacentFacesSize()) return NVCV_ERR_MISMATCH;
  for (unsigned ej = 0; ej < n; ej += 2) {            // 2 adjacencies per edge
    for (unsigned vx = 0; vx < 2; ++vx) {             // for every vertex on the edge
      for (unsigned fc = 0; fc < 2; ++fc) {           // and every face on the edge
        IVF vf;
        vf.vf.vertex = adjVertices[ej + vx];
        vf.vf.face = adjFaces[ej + fc];
        if (vf.vf.vertex && vf.vf.face) {             // if a real vertex and a real face
          --vf.vf.vertex;                             // convert from 1-based index ...
          --vf.vf.face;                               // ... to 0-based index
          topo.push_back(vf);
        }
      }
    }
  }
  std::sort(topo.begin(), topo.end());
  topo.erase(std::unique(topo.begin(), topo.end()), topo.end());
  mesh->resizeDualIndices(unsigned(topo.size()));
  unsigned short *dual     = mesh->getDualIndices(),
                 *numFaces = mesh->getVertexFaceCounts();
  memset(numFaces, 0, mesh->numVertices() * sizeof(*numFaces));
  for (unsigned i = 0; i < topo.size(); ++i) {
    numFaces[topo[i].vf.vertex]++;
    dual[i] = topo[i].vf.face;
  }
  return NVCV_SUCCESS;
}


/********************************************************************************
 * MakeMesh
 ********************************************************************************/

NvCV_Status MakeMesh(const SimpleFaceModelAdapter *fma, GLMesh *mesh) {
  mesh->clear();
  mesh->addVertices(fma->getShapeMeanSize() / 3, const_cast<SimpleFaceModelAdapter*>(fma)->getShapeMean(0));
  mesh->addFaces(fma->getTriangleListSize() / 3, 3, const_cast<SimpleFaceModelAdapter*>(fma)->getTriangleList(0), 0, 0);
  NvCV_Status err = ComputeDualTopologyFromAdjacencies(fma, mesh); // This make vertex normal computation lightning fast
  if (NVCV_SUCCESS != err) return err;
  mesh->computeVertexNormals();
  if (fma->fm.partitions.size()) {
    std::vector<GLMesh::Partition> parts(fma->fm.partitions.size());
    for (unsigned i = unsigned(parts.size()); i--;) {
      const SimpleFaceModel::Partition& fr = fma->fm.partitions[i];
      GLMesh::Partition& to = parts[fr.partitionIndex];
      //to.partitionIndex = fr.partitionIndex;  // to doesn't have a partitionIndex
      to.faceIndex = fr.faceIndex;
      to.numFaces = fr.numFaces;
      to.vertexIndex = fr.vertexIndex;
      to.numVertexIndices = fr.numVertexIndices;
      to.name = fr.name;
      to.materialName = fr.materialName;
      to.smooth = fr.smoothingGroup;
    }
    mesh->partitionMesh(unsigned(parts.size()), parts.data());
  }
  return NVCV_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                            RENDER CONTEXT                            /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define LAMBERTIAN_NUM_LIGHTS 2

class RenderContext {
public:

  RenderContext() {
    m_win = nullptr;
  }

  ~RenderContext() {
    if (m_win) CloseGLContext(m_win);
    m_lam.shutdown();
    m_txr.shutdown();
  }

  NvCV_Status init() {
    if (0 != m_lam.startup())
      return NVCV_ERR_OPENGL;
    if (0 != m_txr.startup())
      return NVCV_ERR_OPENGL;
    return NVCV_SUCCESS;
  }

  void setClearColor(float r, float g, float b, float a = 1.f) { glClearColor(r, g, b, a); }

  void setClearColor(unsigned char r, unsigned char g, unsigned char b) {
    glClearColor(r * (1.f / 255.f), g * (1.f / 255.f), b * (1.f / 255.f), 1.f);
  }

  NvCV_Status makeWindowContext(int wd, int ht, const char *title) {
    NvCV_Status err = MakeGLContext(wd, ht, title, &m_win);
    if (NVCV_SUCCESS == err) {
      m_width = wd;
      m_height = ht;
      glfwMakeContextCurrent(m_win);
      glViewport(0, 0, wd, ht);
      glEnable(GL_DEPTH_TEST);
      glClearColor(0.f, 0.f, 0.f, 1.f);
      if (/*FLAG_orientation*/0) {
        glEnable(GL_CULL_FACE);
        glCullFace((/*FLAG_orientation*/0 > 0) ? GL_BACK : GL_FRONT);
      }
      else {
        glDisable(GL_CULL_FACE);
      }
    }
    return err;
  }

  void computeInverseViewMatrix() {
    #if 0
      Vinv = glm::inverse(V);
    #else // Inversion is simple because we know that it is a rigid transform
      m_Vinv = glm::transpose(m_V);
      m_Vinv[3][0] = -(m_Vinv[0][0] * m_V[3][0] + m_Vinv[1][0] * m_V[3][1] + m_Vinv[2][0] * m_V[3][2]);
      m_Vinv[3][1] = -(m_Vinv[0][1] * m_V[3][0] + m_Vinv[1][1] * m_V[3][1] + m_Vinv[2][1] * m_V[3][2]);
      m_Vinv[3][2] = -(m_Vinv[0][2] * m_V[3][0] + m_Vinv[1][2] * m_V[3][1] + m_Vinv[2][2] * m_V[3][2]);
      m_Vinv[0][3] = 0.f; m_Vinv[1][3] = 0.f;  m_Vinv[2][3] = 0.f; m_Vinv[3][3] = 1.f;
    #endif
  }

  void setViewMatrix(const glm::mat4x4& viewMatrix) {
    m_V = viewMatrix;
    computeInverseViewMatrix();
  }

  void setViewMatrix(const glm::vec3& fromPoint, const glm::vec3& toPoint, const glm::vec3& upVector) {
    m_V = glm::lookAt(fromPoint, toPoint, upVector);
    computeInverseViewMatrix();
  }

  void setOrthoCamera(float hither, float yon) {
    m_P = glm::orthoLH_NO(m_width * -.5f, m_width * +.5f, m_height * -.5f, m_height * +.5f, hither, yon);
  }

  void setOrthoCamera(float wd, float ht, float hither, float yon) {
    m_P = glm::orthoLH_NO(wd * -.5f, wd * +.5f, ht * -.5f, ht * +.5f, hither, yon);
  }

  void setLights(const glm::vec4 locs[LAMBERTIAN_NUM_LIGHTS], const GLSpectrum3f colors[LAMBERTIAN_NUM_LIGHTS]) {
    memcpy(m_lightLoc, locs, sizeof(m_lightLoc));
    memcpy(m_lightColor, colors, sizeof(m_lightColor));
  }

  void setViewOfBound(const GLMesh::BoundingSphere& bsph, const glm::vec3& lookAt, const glm::vec3& up, float vfov,
                      float fracFill, float yDir = 1.f, float z_near = 0.0f, float z_far = 0.0f) {
    float r       = bsph.radius() / fracFill,
          aspect  = (float)m_width / (float)m_height,
          signZ   = -yDir,
          dist;

    if (vfov > 0) {  // Perspective
      dist = r * .5f / tanf(vfov * .5f);
      if (z_near == 0.0f && z_far == 0.0f) {
        // If z_near and z_far are both 0, use default values
        z_near = (dist - r) * 0.2f;
        z_far = (dist + r) * 2.0f;
      }
      m_P = glm::perspective(vfov, aspect, z_near, z_far);
    } else {  // Orthographic
      float   w = r,
              h = r;
      if (aspect < 1.f)   h /= aspect;    // Wide
      else                w *= aspect;    // Tall
      dist = r * 2.f;

      m_P = glm::orthoLH_NO(-w, +w, -h, +h, (dist - r) * signZ, (dist + r) * signZ);
    }
    m_V = glm::lookAt(bsph.center() - glm::normalize(lookAt) * dist, bsph.center(), up);
    computeInverseViewMatrix();
  }

  void setViewOfBound(const GLMesh::BoundingBox& bbox, const glm::vec3& lookAt, const glm::vec3& up, float vfov,
                      float fracFill, float yDir = 1.f, float yOff = 0.f, float z_near = 0.0f, float z_far = 0.0f) {
    glm::vec3 boxSize     = bbox.max() - bbox.min();
    glm::vec3 boxCenter   = bbox.center();
    float     borderFrac  = ((1.f - fracFill) / fracFill),
              dx          = boxSize.x * (1.f + borderFrac),       // border on left and right
              dy          = boxSize.y * (1.f + borderFrac) * (1.f - fabsf(yOff)),
              r           = ((dx > dy) ? dx : dy),                // radius of bounding sphere
              aspectGeom  = dx / dy,
              aspectWind  = (float)m_width / (float)m_height,
              signZ       = -yDir,
              dist;

    if (vfov > 0) {  // Perspective
      dist = r * .5f / tanf(vfov * .5f);
      if (z_near == 0.0f && z_far == 0.0f) {
        // If z_near and z_far are both 0, use default values
        z_near = dist - r;
        z_far = dist + r;
      }
      m_P = glm::perspective(vfov, aspectWind, z_near, z_far);
    } else {  // Orthographic
      if (aspectGeom > aspectWind) dy *= aspectGeom / aspectWind;
      else                         dx *= aspectWind / aspectGeom;
      dist = r * 2.f;
      dx *= .5f;
      dy *= .5f;
      m_P = glm::orthoLH_NO(-dx, +dx, -dy, +dy, (dist - r) * signZ, (dist + r) * signZ);
    }
    boxCenter.y += boxSize.y * yOff;
    m_V = glm::lookAt(boxCenter - glm::normalize(lookAt) * dist, boxCenter, up);
    computeInverseViewMatrix();
  }

  NvCV_Status renderPolyMesh(const GLMesh& mesh, const glm::mat4x4& M, const char *materialOverride = nullptr) {
    NvCV_Status         nvErr = NVCV_SUCCESS;
    glm::mat4x4         VP = m_P * m_V;
    const GLSpectrum3f  defaultDiffuse = { 0.77f, 0.63f, 0.55f },
                        defaultAmbient = defaultDiffuse * 0.3f;

    #ifdef DEBUG_RENDERING
      unsigned why = mesh.notRenderable(0);
      if (why) {
        if (FLAG_debug)
          printf("Mesh %p is not renderable: %s: %s\n", &mesh,
            ((why & GLMesh::NOT_TRIMESH) ? "not a TriMesh" : ""),
            ((why & GLMesh::COMPLEX_TOPOLOGY) ? "complex topology" : "")
          );
        return keErrGeometry;
      }
    #endif // DEBUG_RENDERING

    // Set lights for all shaders
      m_lam.setLights(&m_lightLoc[0].x, m_lightColor[0].data());    // TODO: set this elsewhere

    for (unsigned ix = 0, numPartitions = mesh.numPartitions(); ix < numPartitions; ++ix) {
      GLMesh::Partition pt;
      nvErr = mesh.getPartition(ix, pt);
      BAIL_IF_ERR(nvErr);
      const GLMaterial *mtl = m_mtlLib.getMaterial(materialOverride ? materialOverride : pt.materialName.c_str());

      if (mesh.numNormals()) {  // We should check for textures, too
        const GLSpectrum3f *difColor, *ambColor;
        if (mtl) {
          difColor = &mtl->diffuseColor;
          ambColor = &mtl->ambientColor;
        }
        else {
          difColor = &defaultDiffuse;
          ambColor = &defaultAmbient;
        }
        m_lam.drawTriMesh(mesh.numVertices(), &mesh.getVertices()->x, &mesh.getNormals()->x,
            pt.numVertexIndices, mesh.getVertexIndices() + pt.vertexIndex, &M[0][0], &VP[0][0],
            ambColor->data(), difColor->data());
      }
    }

  bail:
    return nvErr;
  }

  unsigned            m_width, m_height;                    ///< The dimensions of the viewport.
  GLFWwindow          *m_win;                               ///< The window context.
  GLMaterialLibrary   m_mtlLib;                             ///< The material library.
  glm::mat4x4         m_V, m_Vinv;                          ///< The viewing matrix and its inverse.
  glm::mat4x4         m_P;                                  ///< The projection matrix.
  glm::vec4           m_lightLoc[LAMBERTIAN_NUM_LIGHTS];    ///< The light locations.
  GLSpectrum3f        m_lightColor[LAMBERTIAN_NUM_LIGHTS];  ///< The light colors.
  LambertianRenderer  m_lam;                                ///< The Lambertian renderer.
  TextureRenderer     m_txr;                                ///< The texture renderer.
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////                         OPENGL MESH RENDERER                         /////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


class OpenGLMeshRenderer : public MeshRenderer {
public:
  ~OpenGLMeshRenderer();

  static NvCV_Status initDispatch(MeshRenderer::Dispatch *dispatch);
  static NvCV_Status unload();

private:
  OpenGLMeshRenderer();
  SimpleFaceModelAdapter  _sfma;
  RenderContext           _ctx;
  GLMesh                  _mesh;
  glm::vec3               _ctrRot;

  // C-style object-oriented member functions that are usually loaded from DLL, although in this implementation
  // the OpenGLMeshRenderer is compiled directly into the ExpressionApp, and the MeshRendererBroker is
  // automatically adding it to its portfolio of renderers without creating a separate DLL.
  static NvCV_Status create(MeshRenderer **han);
  static void        destroy(MeshRenderer *han);
  static NvCV_Status name(const char **str);
  static NvCV_Status info(const char **str);
  static NvCV_Status read(MeshRenderer *han, const char *modelFile);
  static NvCV_Status init(MeshRenderer *han, unsigned width, unsigned height, const char *windowName);
  static NvCV_Status setCamera(MeshRenderer *han, const float locPt[3], const float lookVec[3], const float upVec[3],
                               float vfov, float near_z = 0.0f, float far_z = 0.0f);
  static NvCV_Status render(MeshRenderer *han,
                      const float exprs[53], const float qrot[4], const float tran[3], NvCVImage *result);
  NvCV_Status        setFOV(float radians, float near_z = 0.0, float far_z = 0.0f);
};

NvCV_Status OpenGLMeshRenderer_InitDispatch(MeshRenderer::Dispatch *dispatch) {
  return OpenGLMeshRenderer::initDispatch(dispatch);
}

NvCV_Status OpenGLMeshRenderer_Unload() {
  return OpenGLMeshRenderer::unload();
}


/********************************************************************************
 * DeformModel
 ********************************************************************************/

static NvCV_Status DeformModel(const SimpleFaceModel& model, const float *identCoeffs, const float *exprCoeffs, GLMesh *mesh) {
  unsigned    size = unsigned(model.shapeMean.size()) * 3,    // the number of floats in the mesh vector
              numCoeffs, i;
  float       *const dst0 = &mesh->getVertices()->x,          // begin
              *const dst1 = dst0 + size;                      // end
  float const *src;
  float       *dst, c;

  memcpy(dst0, model.shapeMean.data(), size * sizeof(*dst0)); // Initialize
  if (identCoeffs) {
    for (i = 0, numCoeffs = unsigned(model.shapeEigenValues.size()), src = model.shapeModes.data()->vec; i < numCoeffs; ++i, ++identCoeffs) {
      if ((c = *identCoeffs) != 0.f) {
        for (dst = dst0; dst != dst1;)
          *dst++ += *src++ * c;
      }
      else {
        src += size;
      }
    }
  }
  for (i = 0, numCoeffs = unsigned(model.blendShapes.size()); i < numCoeffs; ++i, ++exprCoeffs) {
    if ((c = *exprCoeffs) != 0.f) {
      for (dst = dst0, src = model.blendShapes[i].shape.data()->vec; dst != dst1;)
        *dst++ += *src++ * c;
    }
  }
  mesh->computeVertexNormals();
  return NVCV_SUCCESS;
}


OpenGLMeshRenderer::OpenGLMeshRenderer() {
  /*NvCV_Status err =*/ (void)initDispatch(&this->m_dispatch);
}

OpenGLMeshRenderer::~OpenGLMeshRenderer() {
}

NvCV_Status OpenGLMeshRenderer::name(const char **str) {
  static const char name[] = "OpenGL";
  *str = name;
  return NVCV_SUCCESS;
}

NvCV_Status OpenGLMeshRenderer::info(const char **str) {
  static const char info[] = "OpenGL renderer using local illumination";
  *str = info;
  return NVCV_SUCCESS;
}

NvCV_Status OpenGLMeshRenderer::create(MeshRenderer **han) {
  *han = new OpenGLMeshRenderer();
  return NVCV_SUCCESS;
}

void OpenGLMeshRenderer::destroy(MeshRenderer* /*han*/) {
}

NvCV_Status OpenGLMeshRenderer::read(MeshRenderer *han, const char *modelFile) {
  OpenGLMeshRenderer *ren = static_cast<OpenGLMeshRenderer*>(han);
  size_t z = strlen(modelFile);
  if (z < 5) return NVCV_ERR_FILE;
  if (!strcasecmp(".nvf", modelFile + z - 4)) {
    FaceIOErr ioErr = ReadNVFFaceModel(modelFile, &ren->_sfma);  // TODO clear _sfma first
    if (kIOErrNone != ioErr) {
      printf("Error: \"%s\": %s\n", modelFile, FaceIOErrorStringFromCode(ioErr));
      return NVCV_ERR_READ;
    }
    NvCV_Status nvErr;
    nvErr = MakeMesh(&ren->_sfma, &ren->_mesh);
    if (NVCV_SUCCESS != nvErr) return nvErr;

    std::string mtlFile;
    mtlFile.assign(modelFile, 0, strlen(modelFile) - 3);
    mtlFile += "mtl";
    nvErr = ren->_ctx.m_mtlLib.read(mtlFile.c_str());
    unsigned why = ren->_mesh.notRenderable(0);
    if (why) {
      printf("Mesh \"%s\" is not renderable: %s: %s\n", modelFile,
        ((why & GLMesh::NOT_TRIMESH) ? "not a TriMesh" : ""),
        ((why & GLMesh::COMPLEX_TOPOLOGY) ? "complex topology" : "")
      );
      return NVCV_ERR_MISMATCH;
    }
    return NVCV_SUCCESS;
  }
  // else if (!strcasecmp(".obj", file + z - 4)) { read obj files }
  else {
    return NVCV_ERR_FILE;
  }

}

NvCV_Status OpenGLMeshRenderer::init(MeshRenderer *han,
    unsigned width, unsigned height, const char *windowName) {
  OpenGLMeshRenderer        *ren = static_cast<OpenGLMeshRenderer*>(han);
  static const GLSpectrum3f lightColor[LAMBERTIAN_NUM_LIGHTS] = { { 1.f, 1.f, 1.f }, { .8f, .1f, .1f } };
  static const glm::vec4    lightLoc[LAMBERTIAN_NUM_LIGHTS] = { { 0, 0, +1000, 0}, { 100, -200, -500, 0 } };
  NvCV_Status               nvErr;

  nvErr = ren->_ctx.makeWindowContext(width, height, windowName);
  nvErr = ren->_ctx.init();
  ren->_ctx.setClearColor(0.2f, 0.2f, 0.2f, 1.f);
  ren->_ctx.setLights(lightLoc, lightColor);
  ren->setFOV(0.f); // Default orthographic
  return NVCV_SUCCESS;
}

NvCV_Status OpenGLMeshRenderer::setFOV(float fov, float near_z, float far_z) {
  if (0 == _mesh.numVertices())
    return NVCV_ERR_MODEL;
  GLMesh::BoundingBox bbox;
  _mesh.getBoundingBox(&bbox);
  _ctrRot = bbox.center();
  _ctrRot.y = bbox.min().y;  // Assume that assets are designed with Y-up.
  float vShift = (_mesh.numVertices() > 10000) ? 0.15f : 0.0f;  // Heuristic to determine whether there is a neck
  _ctx.setViewOfBound(bbox, glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, +1.f, 0.f), fov, .7f, +1, vShift, near_z, far_z);
  return NVCV_SUCCESS;
}

NvCV_Status OpenGLMeshRenderer::setCamera(MeshRenderer *han, const float locPt[3], const float lookVec[3],
                                          const float upVec[3], float vfov, float near_z, float far_z) {
  if (locPt || lookVec || upVec) {
    if (!locPt || !lookVec || !upVec || vfov <= 0.0f || (near_z == 0.0f && far_z == 0.0f)) return NVCV_ERR_PARAMETER;
    auto &ctx = static_cast<OpenGLMeshRenderer *>(han)->_ctx;
    const float aspect = static_cast<float>(ctx.m_width) / static_cast<float>(ctx.m_height);
    ctx.m_P = glm::perspective(vfov, aspect, near_z, far_z);
    ctx.setViewMatrix(glm::lookAt(glm::make_vec3(locPt), glm::make_vec3(lookVec), glm::make_vec3(upVec)));
    return NVCV_SUCCESS;
  }
  // Default to camera based on bounding box if not enough input arguments are provided
  return static_cast<OpenGLMeshRenderer *>(han)->setFOV(vfov, near_z, far_z);
}

NvCV_Status OpenGLMeshRenderer::render(MeshRenderer *han,
    const float exprs[53], const float qrot[4], const float* trans, NvCVImage *result) {
  OpenGLMeshRenderer *ren = static_cast<OpenGLMeshRenderer*>(han);
  NvCV_Status nvErr;
  glm::mat4x4 M;
  glm::quat q;  // Convert quaternion from {x,y,z,w} --> GLM's {w,x,y,z}

  if (NVCV_RGBA != result->pixelFormat)
    return NVCV_ERR_PIXELFORMAT;

  if (qrot) { q.x = qrot[0]; q.y = qrot[1]; q.z = qrot[2]; q.w = qrot[3]; }
  else      { q.x = 0.0f;    q.y = 0.0f;    q.z = 0.0f;    q.w = 1.0f;    }

  M = glm::mat4_cast(q);
  if (trans) {
    M = glm::translate(glm::mat4x4(1.f), *((const glm::vec3 *)(trans))) * M;
  } else {
    M = glm::translate(glm::mat4x4(1.f), -ren->_ctrRot);
    M = glm::mat4_cast(q) * M;
    M = glm::translate(M, ren->_ctrRot);
  }

  nvErr = DeformModel(ren->_sfma.fm, nullptr, exprs, &ren->_mesh);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  nvErr = ren->_ctx.renderPolyMesh(ren->_mesh, M, NULL);
  glReadPixels(0, 0, result->width, result->height, GL_RGBA, GL_UNSIGNED_BYTE, result->pixels);
  GLenum glErr = glGetError();
  if (glErr)
    return NVCV_ERR_OPENGL;
  // GL returns an image upside-down, but we can use the NvCVImage_FlipY in the caller to flip it with no overhead
  return NVCV_SUCCESS;

}

NvCV_Status OpenGLMeshRenderer::initDispatch(MeshRenderer::Dispatch *dispatch) {
  dispatch->name      = &OpenGLMeshRenderer::name;
  dispatch->info      = &OpenGLMeshRenderer::info;
  dispatch->create    = &OpenGLMeshRenderer::create;
  dispatch->destroy   = &OpenGLMeshRenderer::destroy;
  dispatch->read      = &OpenGLMeshRenderer::read;
  dispatch->init      = &OpenGLMeshRenderer::init;
  dispatch->setCamera = &OpenGLMeshRenderer::setCamera;
  dispatch->render    = &OpenGLMeshRenderer::render;
  return NVCV_SUCCESS;
}

NvCV_Status OpenGLMeshRenderer::unload() {
  return NVCV_SUCCESS;
}
