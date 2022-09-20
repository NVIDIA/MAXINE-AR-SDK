# README
## NVIDIA MAXINE AR SDK: API Source Code and Sample Applications

NVIDIA MAXINE AR SDK offers AI-based, real-time 3D face tracking and modeling, as well as body pose estimation based on a standard web camera feed. Developers can create unique AR effects such as overlaying 3D content on a face, driving 3D characters and virtual interactions in real time. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

The SDK has the following features:

- **Face tracking**, which detects, localizes, and tracks human faces in images or videos by using bounding boxes.
- **Face landmark tracking**, which predicts and tracks the pixel locations of human facial landmark points using 68 or 126 landmark points. The 68 detected facial landmarks follow the Multi-PIE 68 point mark-ups information in [facial point annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/). The 126 landmark points detector can predict more points on the cheeks, the eyes, and on laugh lines. Additionally, it tracks head pose and facial deformation due to head movement and expression in three degrees of freedom in real time.
- **Face mesh**, which reconstructs and tracks a human face via a 3D mesh, as well as its head pose, from the provided facial landmarks.
- **Body Pose Estimation**, which predicts and tracks 34 key points of the human body, with joint angles, in 2D and 3D. It also supports multi-person tracking.
- **Eye contact**, which simulates eye contact by estimating and aligning gaze with the camera to enhance engagement in video communication.
- **Face Expression Estimation**, which estimates face expression (blendshape) coefficients from the provided facial landmarks.

<p align="center">
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_001.png" alt="Face tracking" width="320" height="180"/>
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_002.png" alt="Face landmark tracking - 68 pts" width="320" height="180" />
</p><p align="center">
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_003.png" alt="Face landmark tracking - 126 pts" width="320" height="180"/>
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_004.png" alt="Face mesh" width="320" height="180"/>
</p>
</p><p align="center">
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_005.png" alt="Body Pose estimation" width="480" height="270"/>
</p><p align="center">
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_006.png" alt="Eye contact" width="640" height="237"/>
</p><p align="center">
<img src="https://github.com/NVIDIA/MAXINE-AR-SDK/blob/master/resources/ar_007.png" alt="Face Expression Estimation" width="640" height="175"/>
</p>

The SDK provides four sample applications that demonstrate the features listed above in real time by using a webcam or offline videos.
- **FaceTrack App** which demonstrates the face tracking, landmark tracking and face mesh tracking features.
- **BodyTrack App** which demonstrates the Body Pose estimation feature.
- **GazeRedirect App** which demonstrates the Eye Contact feature.
- **ExpressionApp** which demonstrates the Face Expression Estimation feature.

NVIDIA MAXINE AR SDK is distributed in the following parts:

- This open source repository that includes the [SDK API and proxy linking source code](https://github.com/NVIDIA/MAXINE-AR-SDK/tree/master/nvar), and [sample applications and their dependency libraries](https://github.com/NVIDIA/MAXINE-AR-SDK/tree/master/samples).
- An installer hosted on [NVIDIA Maxine End-user Redistributables page](https://www.nvidia.com/broadcast-sdk-resources) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to [SDK System guide](https://docs.nvidia.com/deeplearning/maxine/ar-sdk-system-guide/index.html) for configuring and integrating the SDK, compiling and running the sample applications. Please visit the [NVIDIA MAXINE AR SDK](https://developer.nvidia.com/maxine-getting-started) webpage for more information about the SDK.

## System requirements
The SDK is supported on NVIDIA GPUs that are based on the NVIDIA® Turing™, Ampere™ or Ada™ architecture and have Tensor Cores.

* Windows OS supported: 64-bit Windows 10 or later
* Microsoft Visual Studio: 2017 (MSVC15.0) or later
* CMake: v3.12 or later
* NVIDIA Graphics Driver for Windows: 511.65 or later

## NVIDIA MAXINE Branding Guidelines
If you integrate an NVIDIA MAXINE SDK within your product, please follow the required branding guidelines that are available [here](
https://www.nvidia.com/maxine-sdk-guidelines)

## Compiling the sample app

### Steps

The open source repository includes the source code to build the sample application, and a proxy file nvARProxy.cpp to enable compilation without explicitly linking against the SDK DLL.

**Note: To download the models and runtime dependencies required by the features, you need to run the [SDK Installer](https://www.nvidia.com/broadcast-sdk-resources).**

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
3.  Use Visual Studio to generate the FaceTrack.exe, BodyTrack.exe, GazeRedirect.exe or ExpressionApp.exe file from the NvAR_SDK.sln file.
*	In CMake, to open Visual Studio, click Open Project.
*	In Visual Studio, select Build > Build Solution.

## Documentation

Please refer to the online documentation guides -
* [NVIDIA AR SDK Programming Guide](https://docs.nvidia.com/deeplearning/maxine/ar-sdk-programming-guide/index.html)
* [NVIDIA AR SDK System Guide](https://docs.nvidia.com/deeplearning/maxine/ar-sdk-system-guide/index.html)
* [NvCVImage API Guide](https://docs.nvidia.com/deeplearning/maxine/nvcvimage-api-guide/index.html)

PDF versions of these guides are also available at the following locations - 
* [NVIDIA AR SDK Programming Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/ar-sdk-programming-guide.pdf)
* [NVIDIA AR SDK System Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/ar-sdk-system-guide.pdf)
* [NvCVImage API Guide](https://docs.nvidia.com/deeplearning/maxine/pdf/nvcvimage-api-guide.pdf)

