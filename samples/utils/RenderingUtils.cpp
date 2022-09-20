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
#include "RenderingUtils.h"
#include "nvAR_defs.h"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"


glm::mat4x4 get_modelview(const NvAR_RenderingParams& rp) {
  glm::quat rotation(rp.rotation.w, rp.rotation.x, rp.rotation.y, rp.rotation.z);
  glm::mat4x4 modelview = glm::mat4_cast(rotation);
  modelview[3][0] = rp.translation.vec[0];
  modelview[3][1] = rp.translation.vec[1];
  return modelview;
};


void draw_wireframe(const cv::Mat& image, const NvAR_FaceMesh& mesh, const NvAR_RenderingParams &rp, cv::Scalar color) {
  glm::mat4x4 modelview  = get_modelview(rp);
  glm::mat4x4 projection = get_projection(rp);
  glm::vec4 viewport = get_opencv_viewport(image.cols, image.rows);
  for (int i = 0; i < mesh.num_triangles; i++) 
  {
    const auto& triangle = mesh.tvi[i];
    const auto p1 =
      glm::project({mesh.vertices[triangle.vec[0]].vec[0], mesh.vertices[triangle.vec[0]].vec[1], mesh.vertices[triangle.vec[0]].vec[2]},
        modelview, projection, viewport);
    const auto p2 =
      glm::project({mesh.vertices[triangle.vec[1]].vec[0], mesh.vertices[triangle.vec[1]].vec[1], mesh.vertices[triangle.vec[1]].vec[2]},
        modelview, projection, viewport);
    const auto p3 =
      glm::project({mesh.vertices[triangle.vec[2]].vec[0], mesh.vertices[triangle.vec[2]].vec[1], mesh.vertices[triangle.vec[2]].vec[2]},
        modelview, projection, viewport);
    if (are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3))) {
      cv::line(image, cv::Point2f(p1.x, p1.y), cv::Point2f(p2.x, p2.y), color);
      cv::line(image, cv::Point2f(p2.x, p2.y), cv::Point2f(p3.x, p3.y), color);
      cv::line(image, cv::Point2f(p3.x, p3.y), cv::Point2f(p1.x, p1.y), color);
    }
  }
};


// Averaging Quaternions, by Markley, Cheng, Crassisdis & Oshman
void average_poses(NvAR_Quaternion *q, unsigned n)
{
  float acc[10];
  memset(acc, 0, sizeof(acc));
  for (NvAR_Quaternion *qEnd = q + n; q != qEnd; ++q) {
    acc[0] += q->x * q->x;              // Compute the normal matrix
    acc[1] += q->x * q->y;
    acc[2] += q->x * q->z;
    acc[3] += q->x * q->w;
    acc[4] += q->y * q->y;
    acc[5] += q->y * q->z;
    acc[6] += q->y * q->w;
    acc[7] += q->z * q->z;
    acc[8] += q->z * q->w;
    acc[9] += q->w * q->w;
  }
  q -= n; // Reset q to its initial value
  cv::Mat M = (cv::Mat_<float>(4, 4) <<
    acc[0], acc[1], acc[2], acc[3],     // Normal matrix
    acc[1], acc[4], acc[5], acc[6],
    acc[2], acc[5], acc[7], acc[8],
    acc[3], acc[6], acc[8], acc[9]);
  cv::Mat r = (cv::Mat_<float>(4, 1) << q->x, q->y, q->z, q->w);
  for (unsigned i = 6; i--; )           // Use power method to get the dominant eigenvector
    r = M * r;                          // It usually converges in 2 iterations, except for sprays > 90 deg.
  float k = r.at<float>(0) * r.at<float>(0)
          + r.at<float>(1) * r.at<float>(1)
          + r.at<float>(2) * r.at<float>(2)
          + r.at<float>(3) * r.at<float>(3);
  if (k) k = 1.f / sqrtf(k);
  q->x = r.at<float>(0) * k;            // Normalize quaternion
  q->y = r.at<float>(1) * k;
  q->z = r.at<float>(2) * k;
  q->w = r.at<float>(3) * k;
}


void set_rotation_from_quaternion(const NvAR_Quaternion *quat, float M[9])
{
  float a, b, c;

  a = quat->x * quat->x;
  b = quat->y * quat->y;
  c = quat->z * quat->z;
  M[0] = 1.f - 2.f * (b + c);
  M[4] = 1.f - 2.f * (a + c);
  M[8] = 1.f - 2.f * (a + b);

  a = quat->x * quat->y;
  b = quat->z * quat->w;
  M[1] = 2.f * (a + b);
  M[3] = 2.f * (a - b);

  a = quat->y * quat->z;
  b = quat->w * quat->x;
  M[5] = 2.f * (a + b);
  M[7] = 2.f * (a - b);

  a = quat->x * quat->z;
  b = quat->w * quat->y;
  M[6] = 2.f * (a + b);
  M[2] = 2.f * (a - b);
}
