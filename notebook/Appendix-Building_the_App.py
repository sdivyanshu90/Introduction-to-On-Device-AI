# %% [markdown]
# Source notebook: Appendix-Building_the_App.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Appendix - Building the App
# 
# Going into the details of the building the final mobile app was slightly outside the scope of this course. We have build this help guide for you to show the main steps and code samples you need to build the app we saw in the last lesson.

# %% [markdown]
# ## TensorFlow Lite
# TensorFlow Lite is a runtime that enables on-device machine learning by helping developers run their models on mobile, embedded, and edge devices. The models produced by TensorFlow Lite work on multiple platform support, covering Android and iOS devices, embedded Linux, and microcontrollers. The toolchain also has diverse language support, which includes Java, Swift, Objective-C, C++, and Python.
# 
# <a href="https://www.tensorflow.org/lite/guide">🔗 &nbsp; TensorFlow Lite guide [ + ]</a>

# %% [markdown]
# ## Delegation
# 
# In the context of TensorFlow Lite, "delegation" refers to the use of delegates to enable hardware acceleration of machine learning models on mobile and edge devices. Delegates act as a bridge between TensorFlow Lite and on-device accelerators like GPUs and DSPs, optimizing performance and efficiency by leveraging specialized hardware capabilities. This process can significantly improve the speed and power consumption of running machine learning models on such devices. For more details, you can visit the TensorFlow Lite Delegates page.
# 
# <a href="https://ai.google.dev/edge/lite/performance/delegates">🔗 &nbsp; TensorFlow Lite Delegates page [ + ]</a>
# 
# ### Qualcomm QNN delegate
# 
# Qualcomm QNN delegate allows you to run models on the NPU.
# 
# <a href="https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.22.6.240515.zip">🔗 &nbsp; Download Qualcomm QNN Delegate – (Zip 724 MB)</a>

# %% [markdown]
# ## End-to-end examples
# 
# You can find end-to-end examples, for common machine learning tasks such as image classification, object detection, pose estimation, question answering, text classification, etc. on multiple platforms. Here is the link to some models we have provided for you.
# 
# <a href="
# https://github.com/quic/ai-hub-models/tree/main/apps/android">🔗 &nbsp; End-to-end examples – GitHub [ + ]</a>
# 
# <b>Models available</b>:
# - ImageClassification
# - ImageSuperResolution
# - SemanticSegmentation

# %% [markdown]
# ## Semantic segmentation code
# The code for the semantic segmentation app we developed in this course is available on Github for you to try.
# 
# <b> Requirements:</b>
# - Java, android-sdk and sdkmanager is already set at user's end
# - User should have Linux QNN SDK in local machine.
# - ANDROID_HOME is set to android-sdk path
# - AI-Hub is properly configured with user token.
# 
# <b>Note:</b> Please execute build_apk.py. This script will compile and download a model from AI-Hub and paste it in your Android Project and Generate app-debug.apk.
# 
# <a href="https://github.com/quic/ai-hub-models/tree/main/apps/android/SemanticSegmentation">🔗 &nbsp; Semantic segmentation code + Guide – GitHub [ + ]</a>

# %% [markdown]
# ## Making the App for iOS
# TensorFlow lite also supports executing models on iOS locally with the same C++ APIs. The models can be run on the Apple NPUs using the CoreML. See the following two links.
# 
# <a href="https://www.tensorflow.org/lite/guide/ios">🔗 &nbsp; TensorFlow lite – iOS quickstart[ + ]</a>
# 
# <a href="https://www.tensorflow.org/lite/performance/coreml_delegate">🔗 &nbsp; Tensorflow Lite – iOS Core ML delegate [ + ]</a>
# 
# <b>Tutorial:</b> The following tutorial shows how to integrate a TensorFlow lite model into iOS for the purposes of on-device deployment.
# 
# <a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-ios/#0">🔗 &nbsp; Tutorial – Recognize Flowers with TensorFlow Lite on iOS
#  [ + ]</a>
# 
# 
# The concepts covered in the course around compilation, profiling, and quantization all apply to iOS as well as Android.
# 
# The main differences between iOS and Android are with respect to Application development and NPU delegation (QNN for Android vs CoreML for iOS).

# %% [markdown]
# ## Course Community
# You can discuss your questions related to course content, notebooks, and building your on-device apps with other course learners in Deeplearning.AI community created for this course.
# 
# <a href="https://learn.deeplearning.ai/courses/introduction-to-on-device-ai/community">🔗 &nbsp; Course Community [ + ]</a>
# 
