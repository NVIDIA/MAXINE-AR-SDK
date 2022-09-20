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

#ifndef __GLSPECTRUM_H
#define __GLSPECTRUM_H


////////////////////////////////////////////////////////////////////////////////
/// The representation used for spectral parameters in surface illumination transport.
////////////////////////////////////////////////////////////////////////////////

struct GLSpectrum3f {
  float r, g, b;  ///< Red, green and blue components of the color spectrum.

  /// Default constructor.
  GLSpectrum3f() {}

  /// Initialization constructor.
  /// @param[in]  R   the  red  component.
  /// @param[in]  G   the green component.
  /// @param[in]  B   the blue  component.
  GLSpectrum3f(float R, float G, float B) { set(R, G, B); }

  /// Access to the array of spectral components.
  /// @return a pointer to the array of spectral components.
  const float* data() const { return &r; }

  /// Access to the array of spectral components.
  /// @return a pointer to the array of spectral components.
  float* data() { return &r; }

  /// Set the spectral components.
  /// @param[in]  R   the  red  component.
  /// @param[in]  G   the green component.
  /// @param[in]  B   the blue  component.
  void set(float R, float G, float B) { r = R; g = G; b = B; };

  /// Componentwise scaling of the spectrum.
  /// @param[in]  the scaling spectrum (RHS).
  /// @return     The LHS, scaled by the RHS.
  GLSpectrum3f& operator*=(const GLSpectrum3f& k) { r *= k.r; g *= k.g; b *= k.b; return *this; }

  /// Componentwise augmentation of the spectrum.
  /// @param[in]  the delta spectrum (RHS).
  /// @return     The LHS, augmented by the RHS.
  GLSpectrum3f& operator+=(const GLSpectrum3f& k) { r += k.r; g += k.g; b += k.b; return *this; }

  /// Scalar scaling of the spectrum.
  /// @param[in]  the scalar (RHS).
  /// @return     The LHS, scaled by the RHS.
  GLSpectrum3f& operator*=(float s) { r *= s;   g *= s;   b *= s;   return *this; }

  /// Scalar scaling of the spectrum.
  /// @param[in]  the scalar (RHS).
  /// @return     The LHS, scaled by the RHS.
  GLSpectrum3f& operator/=(float s) { r /= s;   g /= s;   b /= s;   return *this; }

  /// Componentwise scaling of the spectrum.
  /// @param[in]  k   the scaling spectrum (RHS).
  /// @return     The componentwise product of the LHS and RHS.
  GLSpectrum3f operator*(const GLSpectrum3f& k) const { return GLSpectrum3f(r * k.r, g * k.g, b * k.b); }

  /// Componentwise augmentation of the spectrum.
  /// @param[in]  k   the scale vector (RHS).
  /// @return     The componentwise sum of the LHS and RHS.
  GLSpectrum3f operator+(const GLSpectrum3f& k) const { return GLSpectrum3f(r + k.r, g + k.g, b + k.b); }

  /// Scalar scaling of the spectrum.
  /// @param[in]  s   the scalar (RHS).
  /// @return     The product of the LHS and the RHS scalar.
  GLSpectrum3f operator*(float s) const { return GLSpectrum3f(r * s, g * s, b * s); }

  /// Scalar scaling of the spectrum.
  /// @param[in]  s   the scalar (RHS).
  /// @return     The product of the LHS and the RHS scalar.
  GLSpectrum3f operator/(float s) const { return GLSpectrum3f(r / s, g / s, b / s); }
};

/// Scalar scaling of the spectrum.
/// @param[in]  s   the scalar (LHS).
/// @param[in]  k   the spectrum to be scaled (RHS).
/// @return     The product of the RHS and the scalar LHS.
inline GLSpectrum3f operator*(float s, const GLSpectrum3f& k) { return GLSpectrum3f(s * k.r, s * k.g, s * k.b); }


#endif // __GLSPECTRUM_H
