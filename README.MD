# README
## NVIDIA AR SDK: API Source Code and Sample Applications

NVIDIA AR SDK enables real-time modeling and tracking of human faces from video. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

NVIDIA AR SDK has the following features:

- **Face detection and tracking**, which detects, localizes, and tracks human faces in images or videos by using bounding boxes.
- **Facial landmark detection and tracking**, which predicts and tracks the pixel locations of human facial landmark points and head poses in images or videos. It can predict 68 and 126 landmark points. The 68 detected facial landmarks follow the _Multi-PIE 68 point mark-ups_ information in [facial point annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/). The 126 landmark points detector can predict more points on the cheeks, the eyes, and on laugh lines.
- **Face 3D mesh and tracking**, which reconstructs and tracks a 3D human face and its head pose from the provided facial landmarks.

<div>
<img src="https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/jdsouza-dev/resources/ar_001.png" alt="Face detection and tracking" width="320" height="180"/>
<img src="https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/jdsouza-dev/resources/ar_002.png" alt="Facial landmark detection and tracking - 68 pts" width="320" height="180"/>
</div>
<div>
<img src="https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/jdsouza-dev/resources/ar_003.png" alt="Facial landmark detection and tracking - 126 pts" width="320" height="180"/>
<img src="https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/jdsouza-dev/resources/ar_004.png" alt="Face 3D mesh and tracking" width="320" height="180"/>
</div>


NVIDIA AR SDK provides a sample application that demonstrates the features listed above in real time by using a webcam or offline videos.

NVIDIA AR SDK is distributed in the following parts:

- This open source repository that includes the [SDK API and proxy linking source code](https://github.com/NVIDIA/BROADCAST-AR-SDK/tree/master/nvar), and [sample applications and their dependency libraries](https://github.com/NVIDIA/BROADCAST-AR-SDK/tree/master/samples).
- An installer hosted on [RTX broadcast engine developer page](https://developer.nvidia.com/rtx-broadcast-engine) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to [SDK programming guide](https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/master/NVIDIA%20AR%20SDK%20Programming%20Guide.pdf) for configuring and integrating the SDK, compiling and running the sample applications.

## System requirements
The SDK is supported on NVIDIA GPUs that are based on the NVIDIA® Turing™ architecture. Although the SDK can run on Turing™ GPUs without Tensor Cores, it is optimized for much higher performance on GPUs with Tensor Cores.

* Windows OS supported: 64-bit Windows 10
* Microsoft Visual Studio: 2015 (MSVC14.0) or later
* CMake: v3.12 or later
* NVIDIA Graphics Driver for Windows: 455.57 or later
* NVIDIA CUDA Toolkit: 11.1 or later
* NVIDIA TensorRT: 7.2.0 or later

## NVIDIA Branding Guidelines
If you integrate an NVIDIA Broadcast Engine SDK within your product, please follow the required branding guidelines that are available [here](
https://nvidia.frontify.com/d/uAobRitG8H8B)

## Compiling the sample app

### Steps

The open source repository includes the source code to build the sample application, and a proxy file nvARProxy.cpp to enable compilation without explicitly linking against the SDK DLL.

**Note: To download the models and runtime dependencies required by the features, you need to run the [SDK Installer](https://developer.nvidia.com/rtx-broadcast-engine).**

1.	In the root folder of the downloaded source code, start the CMake GUI and specify the source folder and a build folder for the binary files.
*	For the source folder, ensure that the path ends in OSS.
*	For the build folder, ensure that the path ends in OSS/build.
2.  Use CMake to configure and generate the Visual Studio solution file.
*	Click Configure.
*	When prompted to confirm that CMake can create the build folder, click OK.
*	Select Visual Studio for the generator and x64 for the platform.
*	To complete configuring the Visual Studio solution file, click Finish.
*	To generate the Visual Studio Solution file, click Generate.
*	Verify that the build folder contains the NvAR_SDK.sln file.
3.  Use Visual Studio to generate the FaceTrack.exe file from the NvAR_SDK.sln file.
*	In CMake, to open Visual Studio, click Open Project.
*	In Visual Studio, select Build > Build Solution.

