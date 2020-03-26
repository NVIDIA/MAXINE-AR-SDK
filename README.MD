**Nvidia AR SDK: API Source Code and Sample Applications**

NVIDIA AR SDK enables real-time modeling and tracking of human faces from video. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

NVIDIA AR SDK has the following features:

- **Face detection and tracking**, which detects, localizes, and tracks human faces in images or videos by using bounding boxes.
- **Facial landmark detection and tracking**, which predicts and tracks the pixel locations of human facial landmark points and head poses in images or videos.The detected facial landmarks follow the _Multi-PIE 68 point mark-ups_ information in [facial point annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).
- **Face 3D mesh and tracking**, which reconstructs and tracks a 3D human face and its head pose from the provided facial landmarks **.** NVIDIA AR SDK provides a sample application that demonstrates the features listed above in real time by using a webcam or offline videos.

NVIDIA AR SDK is distributed in the following parts:

-  This open source repository that includes the [SDK API and proxy linking source code](https://github.com/NVIDIA/BROADCAST-AR-SDK/tree/master/nvar), and [sample applications and their dependency libraries](https://github.com/NVIDIA/BROADCAST-AR-SDK/tree/master/samples).
- An installer hosted on [RTX broadcast engine developer page](https://developer.nvidia.com/rtx-broadcast-engine) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to [SDK programming guide](https://github.com/NVIDIA/BROADCAST-AR-SDK/blob/master/NVIDIA%20AR%20SDK%20Programming%20Guide.pdf) for configuring and integrating the SDK, compiling and running the sample applications.
