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

#include "GLMesh.h"

#include <string.h>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////                                 GLMesh                                 ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



// Note: we assume that the transformation is affine, i.e. that M[3] = M[7] = M[11] = 0 and M[15] = 1.
static void TransformPoints(const glm::mat4x4& M, unsigned numPts, const glm::vec3 *pts, glm::vec3 *xPts) {
  for (; numPts--; ++pts, ++xPts) { // NB it is better to do the dot products in double precision
    glm::vec3 q; // Use an intermediate variable to allow transformation in-place.
    q.x = M[0][0] * pts->x + M[1][0] * pts->y + M[2][0] * pts->z + M[3][0];
    q.y = M[0][1] * pts->x + M[1][1] * pts->y + M[2][1] * pts->z + M[3][1];
    q.z = M[0][2] * pts->x + M[1][2] * pts->y + M[2][2] * pts->z + M[3][2];
    *xPts = q;
  }
}


// Note: We assume here that the transformation is isotropic;
// otherwise we would need to transform by the inverse transpose of the upper left.
static void TransformNormals(const glm::mat4x4& M, unsigned numPts, const glm::vec3 *pts, glm::vec3 *xPts) {
  for (; numPts--; ++pts, ++xPts) { // NB it is better to do the dot products in double precision
    glm::vec3 q; // Use an intermediate variable to allow transformation in-place.
    q.x = M[0][0] * pts->x + M[1][0] * pts->y + M[2][0] * pts->z;
    q.y = M[0][1] * pts->x + M[1][1] * pts->y + M[2][1] * pts->z;
    q.z = M[0][2] * pts->x + M[1][2] * pts->y + M[2][2] * pts->z;
    *xPts = glm::normalize(q);
  }
}


////////////////////////////////////////////////////////////////////////////////
// GLMesh API
////////////////////////////////////////////////////////////////////////////////



GLMesh::GLMesh(const GLMesh& mesh) {
  m_faceVertexCount = mesh.m_faceVertexCount;
  m_vertices = mesh.m_vertices;
  m_vertexIndices = mesh.m_vertexIndices;
  m_texCoords = mesh.m_texCoords;
  m_textureIndices = mesh.m_textureIndices;
  m_normals = mesh.m_normals;
  m_normalIndices = mesh.m_normalIndices;
  m_faceNormals = mesh.m_faceNormals;

}

GLMesh::GLMesh() { }
GLMesh::~GLMesh() { }
void        GLMesh::resizeVertices(unsigned n) { m_vertices.resize(n); }
void        GLMesh::resizeTexCoords(unsigned n) {
              m_texCoords.resize(n);
              m_textureIndices.resize(n ? unsigned(m_vertexIndices.size()) : 0);
            }
void        GLMesh::resizeNormals(unsigned n) {
              m_normals.resize(n);
              m_normalIndices.resize(n ? unsigned(m_vertexIndices.size()) : 0);
            }
void        GLMesh::resizeFaces(unsigned n) { m_faceVertexCount.resize(n); }
void        GLMesh::resizeTriangles(unsigned n) { m_faceVertexCount.clear(); m_faceVertexCount.resize(n, 3); }
void        GLMesh::resizeVertexIndices(unsigned n) {
              m_vertexIndices.resize(n);
              m_textureIndices.resize(m_texCoords.size() ? n : 0);
              m_normalIndices.resize(m_normals.size() ? n : 0);
            }
void        GLMesh::resizeDualIndices(unsigned n) {
              m_dualIndices.resize(n);
              m_vertexFaceCount.resize(m_vertices.size());
            }
void        GLMesh::useFaceNormals(bool yes) { m_faceNormals.resize(yes ? numFaces() : 0); }

unsigned    GLMesh::numVertices()   const { return unsigned(m_vertices.size()); }
unsigned    GLMesh::numTexCoords()  const { return unsigned(m_texCoords.size()); }
unsigned    GLMesh::numNormals()    const { return unsigned(m_normals.size()); }
unsigned    GLMesh::numFaces()      const { return unsigned(m_faceVertexCount.size()); }
unsigned    GLMesh::numIndices()    const { return unsigned(m_vertexIndices.size()); }


void GLMesh::initPartitions() {
  m_partitions.resize(1);
  Partition& pt = m_partitions[0];
  pt.faceIndex = 0;
  pt.vertexIndex = 0;
  pt.name.clear();
  pt.materialName.clear();
}


NvCV_Status GLMesh::startPartition(const char *name, const char *material, int smooth) {
  if (name)
    for (const GLMesh::Partition& p : m_partitions)
      if (p.name == name)
        return NVCV_ERR_SELECTOR;

  unsigned i = unsigned(m_partitions.size());
  GLMesh::Partition *pt = &m_partitions[i - 1];

  if (m_faceVertexCount.size() != pt->faceIndex) {
    m_partitions.resize(i + 1);
    pt = &m_partitions[i];
    pt->faceIndex = unsigned(m_faceVertexCount.size());
    pt->vertexIndex = unsigned(m_vertexIndices.size());
  }
  if (name)           pt->name = name;
  if (material)       pt->materialName = material;
  if (smooth >= 0)    pt->smooth = smooth;
  return NVCV_SUCCESS;
}


NvCV_Status GLMesh::partitionMesh(unsigned numPartitions, const GLMesh::Partition *srcPartition) {
  m_partitions.resize(numPartitions);
  for (unsigned i = 0; i < numPartitions; ++i, ++srcPartition) {
    if (srcPartition->faceIndex >= m_faceVertexCount.size()) {
      initPartitions();
      return NVCV_ERR_MISMATCH;
    }

    GLMesh::Partition& pt = m_partitions[i];
    pt.faceIndex = srcPartition->faceIndex;
    pt.vertexIndex = srcPartition->vertexIndex;
    pt.numFaces = srcPartition->numFaces;
    pt.numVertexIndices = srcPartition->numVertexIndices;
    pt.name = srcPartition->name;
    pt.materialName = srcPartition->materialName;
  }
  //computeStartingVertexIndices();

  return NVCV_SUCCESS;
}


NvCV_Status GLMesh::updatePartition(unsigned i, const GLMesh::Partition& update) {
  if (i >= m_partitions.size())
    return NVCV_ERR_FEATURENOTFOUND;

  GLMesh::Partition& pt = m_partitions[i];
  pt.faceIndex = update.faceIndex;
  pt.vertexIndex = update.vertexIndex;
  if (update.name.empty())          pt.name.clear();
  else                              pt.name = update.name;
  if (update.materialName.empty())  pt.materialName.clear();
  else                              pt.materialName = update.materialName;
  return NVCV_SUCCESS;
}


void GLMesh::computeStartingVertexIndices() {
  unsigned vertIx;
  std::sort(m_partitions.begin(), m_partitions.end());
  const unsigned short *faceCount = m_faceVertexCount.data(), *lastFace;
  GLMesh::Partition *pt = m_partitions.data(), *lastPt = pt + m_partitions.size() - 1;
  for (vertIx = 0; pt != lastPt; ++pt) {
    pt->vertexIndex = vertIx;
    for (lastFace = faceCount + (pt[1].faceIndex - pt[0].faceIndex); faceCount != lastFace; ++faceCount)
      vertIx += *faceCount;
  }
  pt->vertexIndex = vertIx;
}


NvCV_Status GLMesh::getPartition(unsigned i, GLMesh::Partition& pt) const {
  if (i > m_partitions.size())
    return NVCV_ERR_FEATURENOTFOUND;
  pt = m_partitions[i];
  return NVCV_SUCCESS;
}


bool GLMesh::indicesMatch(const std::vector<unsigned short>& ivecA, const std::vector<unsigned short>& ivecB) {
  size_t                  n = ivecA.size();
  const unsigned short *a = ivecA.data(),
     *b = ivecB.data();

  if (ivecB.size() != n)
    return false;
  for (; n--; ++a, ++b)
    if (*a != *b)
      return false;
  return true;
}


void GLMesh::assureConsistency() {
  if (m_texCoords.size() && (m_textureIndices.size() != m_vertexIndices.size()))
    m_textureIndices.resize(m_vertexIndices.size());
  if (m_normals.size() && (m_normalIndices.size() != m_vertexIndices.size()))
    m_normalIndices.resize(m_vertexIndices.size());
  if (m_faceNormals.size() && (m_faceNormals.size() != (m_vertexIndices.size() / 3)))
    m_faceNormals.resize(m_vertexIndices.size() / 3);
}


void GLMesh::clear() {
  m_faceVertexCount.resize(0);
  m_vertices.resize(0);
  m_texCoords.resize(0);
  m_normals.resize(0);
  m_faceNormals.resize(0);
  m_vertexIndices.resize(0);
  m_textureIndices.resize(0);
  m_normalIndices.resize(0);
  initPartitions();
}


glm::vec3* GLMesh::getVertices() {
  return m_vertices.data();
}

glm::vec2* GLMesh::getTexCoords() {
  assureConsistency();
  return m_texCoords.size() ? m_texCoords.data() : nullptr;
}

glm::vec3* GLMesh::getNormals() {
  assureConsistency();
  return m_normals.size() ? m_normals.data() : nullptr;
}

glm::vec3* GLMesh::getFaceNormals() {
  assureConsistency();
  return m_faceNormals.size() ? m_faceNormals.data() : nullptr;
}

const glm::vec3* GLMesh::getVertices()    const { return const_cast<GLMesh*>(this)->getVertices(); }
const glm::vec2* GLMesh::getTexCoords()   const { return const_cast<GLMesh*>(this)->getTexCoords(); }
const glm::vec3* GLMesh::getNormals()     const { return const_cast<GLMesh*>(this)->getNormals(); }
const glm::vec3* GLMesh::getFaceNormals() const { return const_cast<GLMesh*>(this)->getFaceNormals(); }

unsigned short* GLMesh::getFaceVertexCounts() { return m_faceVertexCount.data(); }
unsigned short* GLMesh::getVertexIndices() { return m_vertexIndices.data(); }
unsigned short* GLMesh::getTextureIndices() {
  return m_textureIndices.size() ? m_textureIndices.data() : nullptr;
}
unsigned short* GLMesh::getNormalIndices() {
  return m_normalIndices.size() ? m_normalIndices.data() : nullptr;
}
const unsigned short* GLMesh::getFaceVertexCounts() const { return m_faceVertexCount.data(); }
const unsigned short* GLMesh::getVertexIndices()    const { return m_vertexIndices.data(); }
const unsigned short* GLMesh::getTextureIndices()   const { return const_cast<GLMesh*>(this)->getTextureIndices(); }
const unsigned short* GLMesh::getNormalIndices()    const { return const_cast<GLMesh*>(this)->getNormalIndices(); }

unsigned short* GLMesh::getVertexFaceCounts() { return m_vertexFaceCount.data(); }
unsigned short* GLMesh::getDualIndices() {
  return m_dualIndices.size() ? m_dualIndices.data() : nullptr;
}
const unsigned short* GLMesh::getVertexFaceCounts() const { return const_cast<GLMesh*>(this)->getVertexFaceCounts(); }
const unsigned short* GLMesh::getDualIndices() const { return const_cast<GLMesh*>(this)->getDualIndices(); }


void GLMesh::addVertex(float x, float y, float z) { m_vertices.emplace_back(glm::vec3{ x, y, z }); }
void GLMesh::addTexCoord(float u, float v) { m_texCoords.emplace_back(glm::vec2{ u, v }); }
void GLMesh::addNormal(float x, float y, float z) { m_normals.emplace_back(glm::vec3{ x, y, z }); }

void GLMesh::addVertices(unsigned numVertices, const float *vertices) {
  size_t preVertices = m_vertices.size();
  m_vertices.resize(preVertices + numVertices);
  memcpy(m_vertices.data() + preVertices, vertices, numVertices * sizeof(*m_vertices.data()));
}

void GLMesh::addTexCoords(unsigned numTexCoords, const float *texCoords) {
  size_t preTexCoords = m_texCoords.size();
  m_texCoords.resize(preTexCoords + numTexCoords);
  memcpy(m_texCoords.data() + preTexCoords, texCoords, numTexCoords * sizeof(*m_texCoords.data()));
}

void GLMesh::addNormals(unsigned numNormals, const float *normals) {
  size_t preNormals = m_normals.size();
  m_normals.resize(preNormals + numNormals);
  memcpy(m_normals.data() + preNormals, normals, numNormals * sizeof(*m_normals.data()));
}


void GLMesh::addFace(unsigned numVertices, const unsigned short *vertexIndices,
  const unsigned short *textureIndices, const unsigned short *normalIndices)
{
  size_t n;
  m_faceVertexCount.push_back((unsigned short)numVertices);
  if (vertexIndices) {
    n = m_vertexIndices.size();
    m_vertexIndices.resize(n + numVertices);
    memcpy(m_vertexIndices.data() + n, vertexIndices, numVertices * sizeof(*vertexIndices));
  }
  if (textureIndices) {
    n = m_textureIndices.size();
    m_textureIndices.resize(n + numVertices);
    memcpy(m_textureIndices.data() + n, textureIndices, numVertices * sizeof(*textureIndices));
  }
  if (normalIndices) {
    n = m_normalIndices.size();
    m_normalIndices.resize(n + numVertices);
    memcpy(m_normalIndices.data() + n, normalIndices, numVertices * sizeof(*normalIndices));
  }
}


void GLMesh::addFaces(unsigned numFaces, unsigned numVerticesPerFace, const unsigned short *vertexIndices,
  const unsigned short *textureIndices, const unsigned short *normalIndices)
{
  size_t numIndices = numFaces * numVerticesPerFace,
    indexBytes = numIndices * sizeof(*vertexIndices);
  size_t n;

  n = m_faceVertexCount.size();
  m_faceVertexCount.resize(n + numFaces, (unsigned short)numVerticesPerFace);

  if (vertexIndices) {
    n = m_vertexIndices.size();
    m_vertexIndices.resize(n + numIndices);
    memcpy(m_vertexIndices.data() + n, vertexIndices, indexBytes);
  }
  if (textureIndices) {
    n = m_textureIndices.size();
    m_textureIndices.resize(n + numIndices);
    memcpy(m_textureIndices.data() + n, textureIndices, indexBytes);
  }
  if (normalIndices) {
    n = m_normalIndices.size();
    m_normalIndices.resize(n + numIndices);
    memcpy(m_normalIndices.data() + n, normalIndices, indexBytes);
  }
}


bool GLMesh::isTriMesh() const {
  unsigned n;
  const unsigned short *ix;

  for (n = numFaces(), ix = getFaceVertexCounts(); n--; ++ix)
    if (*ix != 3)
      return false;
  return true;
}


bool GLMesh::isQuadMesh() const {
  unsigned n;
  const unsigned short *ix;

  for (n = numFaces(), ix = getFaceVertexCounts(); n--; ++ix)
    if (*ix != 4)
      return false;
  return true;
}


bool GLMesh::isTriQuadMesh() const {
  unsigned n;
  const unsigned short *ix;

  for (n = numFaces(), ix = getFaceVertexCounts(); n--; ++ix)
    if (*ix > 4)
      return false;
  return true;
}


void GLMesh::transform(const glm::mat4x4& M) {
  TransformPoints (M, unsigned(m_vertices.size()),    m_vertices.data(),    m_vertices.data());
  TransformNormals(M, unsigned(m_normals.size()),     m_normals.data(),     m_normals.data());
  TransformNormals(M, unsigned(m_faceNormals.size()), m_faceNormals.data(), m_faceNormals.data());
}


unsigned GLMesh::numPartitions() const {
  return unsigned(m_partitions.size());
}


void GLMesh::finishPartitioning() {
  computeStartingVertexIndices();
}


NvCV_Status GLMesh::setMaterial(const char *name) {
  Partition pt{};
  pt.materialName = name;
  return partitionMesh(1, &pt);
}


void GLMesh::computeFaceNormals(int weighted) {
  const glm::vec3 *vertices = m_vertices.data();
  const unsigned short *numVertices = m_faceVertexCount.data();
  glm::vec3 *nrm, *nrmEnd;
  const unsigned short *ix;
  const glm::vec3 *p0, *p1, *p2;
  glm::vec3 n;
  float mag;

  useFaceNormals(true);
  nrm = getFaceNormals();
  nrmEnd = nrm + numFaces();
  ix = m_vertexIndices.data();

  for (; nrm != nrmEnd; ++nrm, ix += *numVertices++) {
    if (3 == *numVertices) {
      p0 = &vertices[ix[0]];
      p1 = &vertices[ix[1]];
      p2 = &vertices[ix[2]];
      n = glm::cross((*p1 - *p0), (*p2 - *p0));
    }
    else {
      unsigned numPts = *numVertices;
      unsigned i;
      p0 = &vertices[ix[numPts - 1]];
      n = { 0.f, 0.f, 0.f };
      for (i = 0, p0 = &vertices[ix[numPts - 1]]; i < numPts; ++i, p0 = p1) {
        p1 = &vertices[ix[i]];
        n.x -= (p1->y - p0->y) * (p1->z + p0->z);
        n.y -= (p1->z - p0->z) * (p1->x + p0->x);
        n.z -= (p1->x - p0->x) * (p1->y + p0->y);
      }
    }
    mag = glm::length(n);
    if (weighted == 0)       { if (mag) n /= mag; }                // Unit vector
    else if (weighted > 0)   { n *= 0.5f; }                        // Area-weighted normal
    else  /* weighted < 0 */ { if (mag) n /= mag * mag * 0.25f; }  // Inverse-area-weighted vector
    *nrm = n;
  }
}


void GLMesh::computeVertexNormals(int weighted) {
  glm::vec3 nrm;

  computeFaceNormals(weighted);
  if (m_normals.size() != m_vertices.size()) {
    m_normals.resize(m_vertices.size());
    m_normalIndices = m_vertexIndices;
  }

  if (m_vertexFaceCount.size() == m_vertices.size()) {  // We already have the dual topology
    glm::vec3 *n, *nEnd;
    unsigned short *numPolys, *ix, *ixEnd;
    for (nEnd = (n = m_normals.data()) + m_vertexFaceCount.size(), numPolys = m_vertexFaceCount.data(), ix = m_dualIndices.data(); n != nEnd; ++n, ++numPolys) {
      for (ixEnd = ix + *numPolys, nrm = { 0.f, 0.f, 0.f }; ix != ixEnd; ++ix)
        nrm += m_faceNormals[*ix];
      *n = glm::normalize(nrm);
    }
  }
}


void GLMesh::BoundingBox::unionPoint(const glm::vec3& pt) {   /* This works with NaN's */
  if (!(_box[0].x < pt.x))  _box[0].x = pt.x;     if (!(_box[1].x > pt.x))  _box[1].x = pt.x;
  if (!(_box[0].y < pt.y))  _box[0].y = pt.y;     if (!(_box[1].y > pt.y))  _box[1].y = pt.y;
  if (!(_box[0].z < pt.z))  _box[0].z = pt.z;     if (!(_box[1].z > pt.z))  _box[1].z = pt.z;
}

void GLMesh::BoundingBox::set(unsigned numPts, const glm::vec3 *pts) {
  _box[0] = pts[0];
  _box[1] = pts[0];
  for (++pts; --numPts; ++pts)
    unionPoint(*pts);
}

void GLMesh::BoundingBox::set(unsigned numPts, const glm::vec3 *pts, const glm::mat4x4& M) {
  memset(this, -1, sizeof(*this));    // Set to NaN
  for (; numPts--; ++pts) {
    glm::vec3 q;
    TransformPoints(M, 1, pts, &q);
    unionPoint(q);
  }
}

void GLMesh::BoundingSphere::set(unsigned numPts, const glm::vec3 *pts) {
  /* Ritter algorithm */
  float d, d0;
  glm::vec3 p0, p1;
  const glm::vec3 *pp, *pEnd = pts + numPts;

  p0 = p1 = pts[0];                             // Choose one point
  for (pp = pts + 1, d0 = 0; pp != pEnd; ++pp) {
    if (!(d0 > (d = glm::distance(p0, *pp)))) {
      d0 = d;
      p1 = *pp;                                 // Find the furthest point
    }
  }
  p0 = p1;                                      // Choose that furthest point
  for (pp = pts, d0 = 0; pp != pEnd; ++pp) {
    if (!(d0 > (d = glm::distance(p0, *pp)))) {
      d0 = d;
      p1 = *pp;                                 // Find the furthest point from that
    }
  }

  _radius = d0 * .5f;                           // Make a sphere ...
  _center = (p1 - p0) * .5f + p0;               // ... from these furthest points

  bool done;
  // Accommodate every outlier as we encounter them
  do {
    done = true;
    for (pp = pts; pp != pEnd; ++pp) {          // Check that all points are in this sphere
      glm::vec3 v = *pp - _center;
      d0 = glm::length(v);
      if (d0 > _radius) {                       // If not, ...
        d = (d0 - _radius) * .5f;
        _center += v * (d / d0);                // ... adjust the sphere center ...
        _radius += d;                           // ... and radius to accommodate this new point
        done = false;
      }
    }
  } while (!done);
}


void GLMesh::BoundingSphere::set(unsigned numPts, const glm::vec3 *pts, const glm::mat4x4& M) {
  std::vector<glm::vec3> xPts(numPts);
  TransformPoints(M, numPts, pts, xPts.data());
  set(numPts, xPts.data());
}

void GLMesh::getBoundingBox(BoundingBox *bbox, const glm::mat4x4 *M) const {
  if (M) bbox->set(unsigned(m_vertices.size()), m_vertices.data(), *M);
  else   bbox->set(unsigned(m_vertices.size()), m_vertices.data());
}


void GLMesh::getBoundingSphere(BoundingSphere *bsph, const glm::mat4x4 *M) const {
  if (M)  bsph->set(unsigned(m_vertices.size()), m_vertices.data(), *M);
  else    bsph->set(unsigned(m_vertices.size()), m_vertices.data());
}


unsigned GLMesh::notRenderable(unsigned /*options*/) const {
  unsigned result = RENDERABLE;

  if (!isTriMesh())
    result |= NOT_TRIMESH;
  if (0 != m_textureIndices.size() && !indicesMatch(m_vertexIndices, m_textureIndices))
    result |= COMPLEX_TOPOLOGY;
  if (0 != m_normalIndices.size() && !indicesMatch(m_vertexIndices, m_normalIndices))
    result |= COMPLEX_TOPOLOGY;

  return result;
}


NvCV_Status GLMesh::append(const GLMesh& other, const glm::mat4x4 *M) {
  if ((!numTexCoords() != !other.numTexCoords()) || (!numNormals() != !other.numNormals()))
    return NVCV_ERR_MISMATCH;

  unsigned indexOffset = numVertices(),
    thisCount = numIndices(),
    otherCount = other.numIndices(),
    i;

  // Add vertices
  addVertices(other.numVertices(), &other.getVertices()->x);
  if (M)
    TransformPoints(*M, other.numVertices(), getVertices() + indexOffset, getVertices() + indexOffset);
  if (0 != (i = other.numTexCoords()))
    addTexCoords(i, &other.getTexCoords()->x);
  if (0 != (i = other.numNormals())) {
    addNormals(i, &other.getNormals()->x);
    if (M)
      TransformNormals(*M, other.numNormals(), getNormals() + indexOffset, getNormals() + indexOffset);
  }

  {   // Add indices
    const unsigned short *nvx = other.getFaceVertexCounts(),
       *vix = other.getVertexIndices(),
       *tix = other.getTextureIndices(),
       *nix = other.getNormalIndices();
    for (i = other.numFaces(); i--; ++nvx) {
      addFace(*nvx, vix, tix, nix);
      vix += *nvx;
      if (tix) tix += *nvx;
      if (nix) nix += *nvx;
    }
  }


  {   // Offset the new indices
    unsigned short *ix;
    for (i = otherCount, ix = getVertexIndices() + thisCount; i--; ++ix)
      *ix += (unsigned short)indexOffset;
    if (nullptr != (ix = getTextureIndices()))
      for (i = otherCount, ix += thisCount; i--; ++ix)
        *ix += (unsigned short)indexOffset;
    if (nullptr != (ix = getNormalIndices()))
      for (i = otherCount, ix += thisCount; i--; ++ix)
        *ix += (unsigned short)indexOffset;
  }

  return NVCV_SUCCESS;
}
