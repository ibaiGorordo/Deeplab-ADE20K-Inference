# Deeplab ADE20K inference (Tensorflow 2)
Python program to visualize the results from the Deeplab model (trained on **ADE20K dataset**).

## Required Python packages
* **numpy**: For the array calculations.
* **tensorflow**: Tensorflow 2.
* **Pillow**: To read the image.
* **Matplotlib**: To plot the results.

## HOW TO USE
* Download the required packages.
* Download the model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). **ONLY model trained in ADE20K**
* Change the name of the path to the model (model_path) and the image (img_path) you want to test.
* Run the program.

## Example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibaiGorordo/Deeplab-ADE20K-Inference/blob/master/DeepLab_ADE20K_inference_Demo.ipynb)


For an example on how to visualize the inference of DeepLab model (ADE20K) run **inferenceDeeplabADE20K.py** script.

```
git clone https://github.com/ibaiGorordo/Deeplab-ADE20K-Inference.git
python inferenceDeeplabADE20K.py
```

![Deeplab ADE20K inference example](https://github.com/ibaiGorordo/Deeplab-ADE20K-Inference/blob/master/output.png)

