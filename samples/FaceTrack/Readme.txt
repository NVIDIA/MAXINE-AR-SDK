FaceTrack is a sample Windows application that demonstrates the face tracking, landmark tracking,
and 3D mesh tracking features of the NVIDIA AR SDK. The application requires a video feed
from a camera connected to the computer running the application, or from a video file, as specified
with command-line arguments (enumerated by executing: FaceTrack.exe --help). 

The sample application can run in 3 modes which can be toggled through the '1', '2' and '3' keys on 
the keyboard
1 - Face tracking
2 - Facial landmark tracking
3 - 3D mesh tracking

For more controls and configurations of the sample app, please read the SDK programming guide.

3D mesh tracking feature requires a 3D Morphable Face Model (3DMM). 
NVIDIA AR SDK does not include a 3DMM. Therefore, if you are using the 3D mesh tracking feature, you must configure NVIDIA AR SDK with 3DMM. 
Note: To configure NVIDIA AR SDK with 3DMM, you can use the Surrey Face Model, as described in the steps below, or your own model.
The sample app can still be used in mode 1 and 2 without this file.

NVIDIA AR SDK provides the ConvertSurreyFaceModel.exe utility to convert 3DMM files to the NVIDIA .nvf format that is required by the SDK.
To generate the face model file face_model0.nvf required for 3D mesh tracking:

1) Download the following Surrey Face Model files from the eos project page on Github:
	sfm_shape_3448.bin
	expression_blendshapes_3448.bin
	sfm_3448_edge_topology.json
	sfm_model_contours.json
	ibug_to_sfm.txt

2) Convert the files that you downloaded to the NVIDIA .nvf format.

tools\ConvertSurreyFaceModel.exe
--shape=<path>/sfm_shape_3448.bin 
--blend_shape=<path>/expression_blendshapes_3448.bin
--topology=<path>/sfm_3448_edge_topology.json 
--contours=<path>/sfm_model_contours.json 
--ibug=<path>/ibug_to_sfm.txt
--out=<output-path>/face_model0.nvf

path
The full or relative path to the folder that contains the Surrey Face Model files that you downloaded.
Either the forward slash (/) or back slash (\) can be used as a separator between path elements.

output-path
The full or relative path to the folder where you want the output .nvf format file to be written.
Either the forward slash (/) or back slash (\) can be used as a separator between path elements.

The ConvertSurreyFaceModel.exe file is distributed in the https://github.com/nvidia/BROADCAST-AR-SDK repo.

3) The sample application provided with NVIDIA AR SDK requires that the model file be named face_model0.nvf.
Place the face_model0.nvf file in the /bin/models folder.
