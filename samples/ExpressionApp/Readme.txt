ExpressionApp is a sample application using the AR SDK to extract face expression signals from video. These signals are
used to control the expressions, pose and gaze of a 3D morphable face model. The application can either process
real-time video from a webcam or offline videos from files. It illustrates the facial keypoints that are tracked, plots
the expression signals that are derived, and renders an animated 3D avatar mesh.

The application runs either the Face3DReconstruction, or FaceExpression feature, depending on which expression mode is
used. The expression mode is toggled using the '1' and '2' keys on the keyboard
1 - Face3DReconstruction expression estimation
2 - FaceExpression expression estimation (default, and recommended for avatar animation)

The FaceExpression mode is preferred for avatar animation. Note that Face3DReconstruction is demonstrated for its
ability to track the face over time for AR effects. This feature enables identity face shape estimation on top of
expression estimation and is better demonstrated in the FaceTrack sample application. The resulting expression weights
from FaceExpression is more accurate than from Face3DReconstruction.

For details on command line arguments, execute ExpressionApp.exe --help.
For more controls and configurations of the sample app, including expression definition and conversion to ARKit
blendshapes, please read the SDK programming guide. It also contains information about how to control the GUI which can
be enabled by running the application with the --show_ui argument.