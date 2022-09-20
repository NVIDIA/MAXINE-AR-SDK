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

#ifndef __GLMESH_H
#define __GLMESH_H

#include <string>
#include <vector>

#include "glm/glm.hpp"
#include "nvCVStatus.h"


class GLMesh {
public:
  struct Partition {
    unsigned    faceIndex;        ///< The index of the first face in the partition.
    unsigned    numFaces;         ///< The number of faces in the partition.
    unsigned    vertexIndex;      ///< The index of the first topological vertex in the partition.
    unsigned    numVertexIndices; ///< The number of topological vertices in the partition.
    std::string name;             ///< The name of the partition.
    std::string materialName;     ///< The name of the material assigned to the partition.
    int         smooth;           ///< The smoothing group > 0; no smoothing == 0; unassigned < 0.

    Partition() { smooth = -1; }
    bool operator<(const Partition& pt) const { return faceIndex < pt.faceIndex; }
    void finishPartitioning();
  };
  class BoundingBox {
  public:
    void              unionPoint(const glm::vec3& pt);
    void              set(unsigned numPts, const glm::vec3* pts);
    void              set(unsigned numPts, const glm::vec3* pts, const glm::mat4x4& M);
    glm::vec3&        min()           { return _box[0]; }
    glm::vec3&        max()           { return _box[1]; }
    const glm::vec3&  min()     const { return _box[0]; }
    const glm::vec3&  max()     const { return _box[1]; }
    glm::vec3         center()  const { return (_box[0] + _box[1]) * 0.5f; }
  private:
    glm::vec3         _box[2];
  };
  class BoundingSphere {
  public:
    void              set(unsigned numPts, const glm::vec3* pts);
    void              set(unsigned numPts, const glm::vec3* pts, const glm::mat4x4& M);
    glm::vec3&        center()        { return _center; }
    const glm::vec3&  center() const  { return _center; }
    float&            radius()        { return _radius; }
    float             radius() const  { return _radius; }

  private:
    glm::vec3         _center;
    float             _radius;
  };
  enum { BOUNDARY = 0xFFFFu };        ///< An index that indicates the boundary

  GLMesh();
  GLMesh(const GLMesh& mesh);
  ~GLMesh();


  /// Get the number of faces.
  /// @return the number of faces.
  unsigned                numFaces()              const;

  /// Get the number of XYZ vertices.
  /// @return the number of XYZ vertices.
  unsigned                numVertices()           const;

  /// Get the number of UV texture coordinates.
  /// @return the number of UV texture coordinates.
  unsigned                numTexCoords()          const;

  /// Get the number of XYZ normals.
  /// @return the number of XYZ normals.
  unsigned                numNormals()            const;

  /// Get the number of vertex indices.
  /// @return the number of vertex indices.
  unsigned                numIndices()            const;

  /// Evaluate whether the mesh is composed only of triangles.
  /// @return true    if all faces have 3 vertices; false otherwise.
  bool                    isTriMesh()             const;

  /// Evaluate whether the mesh is composed only of quadrilaterals.
  /// @return true    if all faces have 4 vertices; false otherwise.
  bool                    isQuadMesh()            const;

  /// Evaluate whether the mesh is composed only of triangles and quadrilaterals.
  /// @return true    if no face has greater than 4 vertices; false otherwise.
  bool                    isTriQuadMesh()         const;

  void                    resizeVertices(unsigned numVert);
  void                    resizeTexCoords(unsigned numTexCoord);
  void                    resizeNormals(unsigned numNorm);
  void                    resizeFaces(unsigned numFace);
  void                    resizeTriangles(unsigned numTriangles);
  void                    resizeVertexIndices(unsigned numIndices);
  void                    resizeDualIndices(unsigned numIndices);
  void                    clear();

  glm::vec3*              getVertices();                  ///< Get the vertices. @return a pointer to the vertices.
  const glm::vec3*        getVertices()           const;  ///< Get the vertices. @return a pointer to the vertices.
  glm::vec2*              getTexCoords();                 ///< Get the texture coordinates. @return a pointer to the texture coordinates.
  const glm::vec2*        getTexCoords()          const;  ///< Get the texture coordinates. @return a pointer to the texture coordinates.
  glm::vec3*              getNormals();                   ///< Get the vertex normals. @return a pointer to the vertex normals.
  const glm::vec3*        getNormals()            const;  ///< Get the vertex normals. @return a pointer to the vertex normals.
  glm::vec3*              getFaceNormals();               ///< Get the face normals, computed with computeFaceNormals(). @return a pointer to the vertex normals.
  const glm::vec3*        getFaceNormals()        const;  ///< Get the face normals, computed with computeFaceNormals(). @return a pointer to the vertex normals.
  unsigned short*         getFaceVertexCounts();          ///< Get the vertex counts for each face, in the primal topology. @return an array of vertex counts, one per face.
  const unsigned short*   getFaceVertexCounts()   const;  ///< Get the vertex counts for each face, in the primal topology. @return an array of vertex counts, one per face.
  unsigned short*         getVertexIndices();             ///< Get the vertex indices for each face: the primal topology. @return a pointer to the vertex indices.
  const unsigned short*   getVertexIndices()      const;  ///< Get the vertex indices for each face: the primal topology. @return a pointer to the vertex indices.
  unsigned short*         getTextureIndices();            ///< Get the texture indices for each face: the primal topology. @return a pointer to the texture indices.
  const unsigned short*   getTextureIndices()     const;  ///< Get the texture indices for each face: the primal topology. @return a pointer to the texture indices.
  unsigned short*         getNormalIndices();             ///< Get the normal indices for each face: the primal topology. @return a pointer to the normal indices.
  const unsigned short*   getNormalIndices()      const;  ///< Get the normal indices for each face: the primal topology. @return a pointer to the normal indices.
  unsigned short*         getVertexFaceCounts();          ///< Get the face counts for each vertex, in the dual topology. @return an array of face counts, one per vertex.
  const unsigned short*   getVertexFaceCounts()   const;  ///< Get the face counts for each vertex, in the dual topology. @return an array of face counts, one per vertex.
  unsigned short*         getDualIndices();               ///< Get the face indices for each vertex: the dual topology. @return a pointer to the dual face indices.
  const unsigned short*   getDualIndices()        const;  ///< Get the face indices for each vertex: the dual topology. @return a pointer to the dual face indices.

  void                    addVertex(float x, float y, float z);
  void                    addTexCoord(float u, float v);
  void                    addNormal(float x, float y, float z);

  void                    addVertices(unsigned numVertices, const float* vertices);
  void                    addTexCoords(unsigned numTexCoords, const float* texCoords);
  void                    addNormals(unsigned numNormals, const float* normals);

  void                    addFace(unsigned numVertices, const unsigned short* vertexIndices,
                              const unsigned short* textureIndices, const unsigned short* normalIndices);

  void                    addFaces(unsigned numFaces, unsigned numVerticesPerFace, const unsigned short* vertexIndices,
                              const unsigned short* textureIndices, const unsigned short* normalIndices);

  /// Compute the normals per face.
  /// @param[in]  specify the weighing for the normals. In all cases, the zero vector will
  ///             be returned for faces with zero area.
  ///             0:  unit vectors.
  ///             +1: vectors weighted by the area.
  ///             -1: vectors weighted by the reciprocal of the area.
  void                    computeFaceNormals(int weighted = 0);

  /// Compute the vertex normals. The face normals will be computed in the process.
  /// @param[in]  weighted  Determines the weighting used to combine the face normals:
  ///                       0: all incident faces normals will have the same weight.
  ///                       -1: the normals will be weighted by inverse area of the face.
  void                    computeVertexNormals(int weighted = 0);

  void                    transform(const glm::mat4x4& M);

  /// Get the number of partitions.
  /// There is always at least one, which may neither have a name nor a material.
  /// @return the number of partitions.
  unsigned                numPartitions() const;

  /// The easiest way to partition a mesh: call this after all vertices and attributes
  /// are recorded, and before the first face of each partition is recorded.
  /// @param[in]  name        the name of the new partition.
  /// @param[in]  material    the name of the material to be used in the new partition.
  /// @param[in]  smooth      {-1, 0, 1} means {unspecified, not smooth, smooth}.
  /// @return     NvCV_StatusNone       if the partition was retrieved successfully.
  /// @return     NvCV_StatusDuplicate  if a partition with the same name already exists.
  NvCV_Status             startPartition(const char* name, const char* material, int smooth = -1);

  /// Get the specified partition.
  /// @param[in]  i   the index of the partition to retrieve.
  /// @param[out] pt  a place to store the specified partition.
  /// @return     NvCV_StatusNone       if the partition was retrieved successfully.
  /// @return     NvCV_StatusTooBig     if the face index was >= the number of faces.
  NvCV_Status             getPartition(unsigned i, Partition& pt) const;

  /// Update the specified partition.
  /// The function finishPartitioning() should be called
  /// after the last updatePartition() has been called.
  /// @param[in]  i   the index of the partition.
  /// @param[in]  partition   the desired value for the specified partition.
  /// @return     NvCV_StatusNone       if the partition was updated successfully.
  /// @return     NvCV_StatusTooBig     if the face index was >= the number of faces.
  NvCV_Status             updatePartition(unsigned i, const Partition& partition);

  /// Partition the mesh.
  /// @param[in]  numPartitions   the number of partitions.
  /// @param[in]  partitions      the array of partitions. Only { faceIndex, name, and
  ///                             materialName need be supplied}; the rest are computed.
  /// @return     NvCV_StatusNone       if the partition was executed successfully.
  /// @return     NvCV_StatusTooBig     if any faceIndex was >= the number of faces.
  NvCV_Status             partitionMesh(unsigned numPartitions, const Partition* partitions);

  /// The last step after partitioning with updatePartition().
  /// This is not needed if the partitions were created solely with the use of
  /// startPartition() or PartitionMesh().
  /// @note The partitions may be reordered (sorted) after calling finishPartitioning().
  void                    finishPartitioning();

  /// Set a single material for the whole mesh.
  /// @param[in]  name    the name of the material.
  NvCV_Status             setMaterial(const char* name);

  /// Get the bounding box, optionally with an affine transformation.
  /// @param[out] bbox    a place to store the bounding box.
  /// @param[in]  M       pointer to a modeling matrix; NULL implies the identity.
  void                    getBoundingBox(BoundingBox* bbox, const glm::mat4x4* M = nullptr) const;

  /// Get the bounding box, optionally with an affine transformation.
  /// @param[out] bbox    a place to store the bounding box.
  /// @param[in]  M       pointer to a modeling matrix; NULL implies the identity.
  void                    getBoundingSphere(BoundingSphere* bsph, const glm::mat4x4* M = nullptr) const;

  /// Query whether the PolyMesh is not renderable easily by Open GL.
  /// Since the more typical query would be whether it is renderable instead,
  /// this seems like negative logic, but this choice was made to return a bit vector
  /// indicating the reason that the Polymesh is not renderable.
  /// @param[in]  options Rendering options; currently ignored.
  /// @return     RENDERABLE       if the PolyMesh is renderable. Otherwise a bit vector of:
  ///             NOT_TRIMESH      if some faces are not triangular;
  ///             COMPLEX_TOPOLOGY if the vertex attribute topology is inconsistent;
  unsigned                notRenderable(unsigned options) const;

  /// Append another mesh.
  /// @param[in]  mesh    the other mesh.
  /// @param[in]  M       an optional affine transform
  NvCV_Status             append(const GLMesh& mesh, const glm::mat4x4* M = nullptr);

  /// Bit vector components indicating non-renderability.
  enum {
    RENDERABLE        = 0x0,  ///< The PolyMesh is renderable.
    NOT_TRIMESH       = 0x1,  ///< Some faces are not triangular.
    COMPLEX_TOPOLOGY  = 0x2   ///< The vertex topology is not consistent.
  };

private:
  void        initPartitions();
  void        computeStartingVertexIndices();
  static bool indicesMatch(const std::vector<unsigned short>& ivecA, const std::vector<unsigned short>& ivecB);
  void        assureConsistency();
  void        useFaceNormals(bool yes);

  std::vector<unsigned short> m_faceVertexCount;

  std::vector<glm::vec3>      m_vertices;
  std::vector<unsigned short> m_vertexIndices;

  std::vector<glm::vec2>      m_texCoords;
  std::vector<unsigned short> m_textureIndices;

  std::vector<glm::vec3>      m_normals;
  std::vector<unsigned short> m_normalIndices;

  std::vector<glm::vec3>      m_faceNormals;

  std::vector<Partition>      m_partitions;

  std::vector<unsigned short> m_vertexFaceCount;  // the number of faces surrounding each vertex
  std::vector<unsigned short> m_dualIndices;      // the face indices for each vertex
};

#endif // __GLMESH_H
