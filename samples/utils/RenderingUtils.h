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
#ifndef __RENDERING_UTILS__
#define __RENDERING_UTILS__

#include "nvAR_defs.h"
#include "glm/gtc/matrix_transform.hpp"
#include "opencv2/opencv.hpp"


/** Get the model*view matrix from the rendering parameters.
 * @param[in] rp  the input rendering parameters.
 * @return        the corresponding model*view matrix.
 */
glm::mat4x4 get_modelview(const NvAR_RenderingParams& rp);

/** Get the projection matrix from the rendering parameters.
* @param[in] rp  the input rendering parameters.
* @return        the corresponding projection matrix.
*/
inline glm::mat4x4 get_projection(const NvAR_RenderingParams& rp) {
    return glm::ortho<float>(rp.frustum.left, rp.frustum.right, rp.frustum.bottom, rp.frustum.top);
};

/**
 * @brief Returns a glm/OpenGL compatible viewport vector that flips y and
 * has the origin on the top-left, like in OpenCV.
 */
inline glm::vec4 get_opencv_viewport(int width, int height) { return glm::vec4(0, height, width, -height); };

/**
 * Computes whether the triangle formed out of the given three vertices is
 * counter-clockwise in screen space. Assumes the origin of the screen is on
 * the top-left, and the y-axis goes down (as in OpenCV images).
 *
 * @param[in] v0 First vertex.
 * @param[in] v1 Second vertex.
 * @param[in] v2 Third vertex.
 * @return Whether the vertices are CCW in screen space.
 */
template <typename T, glm::precision P = glm::defaultp>
bool are_vertices_ccw_in_screen_space(const glm::tvec2<T, P>& v0, const glm::tvec2<T, P>& v1,
                                      const glm::tvec2<T, P>& v2) {
  const auto dx01 = v1[0] - v0[0];  // todo: replace with x/y (GLM)
  const auto dy01 = v1[1] - v0[1];
  const auto dx02 = v2[0] - v0[0];
  const auto dy02 = v2[1] - v0[1];

  return (dx01 * dy02 - dy01 * dx02 <
          T(0));  // Original: (dx01*dy02 - dy01*dx02 > 0.0f). But: OpenCV has origin top-left, y goes down
};

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] color Colour of the mesh to be drawn.
 */
void draw_wireframe(const cv::Mat& image, const NvAR_FaceMesh& mesh, const NvAR_RenderingParams &rp, cv::Scalar color = cv::Scalar(0, 255, 0, 255));


/** Averaging Quaternions, by Markley, Cheng, Crassisdis & Oshman.
 * @param[in,out] q   on input, an array of quaternions of length n.
 *                    on output, the average of the input quaternions.
 * @param[in]     n   the number ofg quaternions to be averaged.
 */
void average_poses(NvAR_Quaternion *q, unsigned n);


/** Set the 3x3 rotation matrix from a quaternion.
 * @param[in]   the input normalized quaternion.
 * @param[out]  the corresponding 3x3 rotation matrix.
 */
void set_rotation_from_quaternion(const NvAR_Quaternion *quat, float M[3*3]);


#endif // __RENDERING_UTILS__