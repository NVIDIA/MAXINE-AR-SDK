**Nvidia AR SDK: API Source Code and Sample Applications**

NVIDIA AR SDK enables real-time modeling and tracking of human faces from video. The SDK is powered by NVIDIA graphics processing units (GPUs) with Tensor Cores, and as a result, the algorithm throughput is greatly accelerated, and latency is reduced.

NVIDIA AR SDK has the following features:

- **Face detection and tracking**, which detects, localizes, and tracks human faces in images or videos by using bounding boxes.
- **Facial landmark detection and tracking**, which which predicts and tracks the pixel locations of human facial landmark points and head poses in images or videos.The detected facial landmarks follow the _Multi-PIE 68 point mark-ups_ information in [Facial point annotations](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).
- **Face 3D mesh and tracking**, which reconstructs and tracks a 3D human face and its head pose from the provided facial landmarks **.** NVIDIA AR SDK provides a sample application that demonstrates the features listed above n real time by using a webcam or offline videos.

NVIDIA AR SDK is distributed in the following parts:

-  This open source repository that includes the SDK API and proxy linking source code (_add link to the &quot;nvar&quot; folder in this repo_), sample applications and their dependency libraries (_add link to the &quot;samples&quot; folder in this repo)_.
- An Installer hosted on Nvidia Dev Zone (_link to be added_) that installs the SDK DLLs, the models, and the SDK dependency libraries.

Please refer to SDK programming guide (_add link to programming guide in this repo_) for configuring the SDK, integrating the SDK, compiling and running the sample application.
