# Introduction
I believe that one of the most interesting aspects of deep learning is the design of network architectures for computer vision applications. Since 2012, many architectures have been proposed. While some building blocks have become standard, the overall structure of these networks often looks drastically different from one another.

Some architectures appear simple, others complex, and some, let's say, simply original! These differences highlight the creative process often involved in developing network architectures for computer vision applications. Thanks to modern GPUs available in consumer computers, it's now quite feasible to design your own architecture, experiment with ideas, and start training from scratch.

# "Patches Are All You Need?" - ConvMixer Network
Recently, I came across the paper ["Patches are all you need?"](https://arxiv.org/abs/2201.09792) and found it to be quite original. In the past few years, transformer architectures have gained popularity in computer vision, often achieving excellent results but demanding more resources than traditional CNNs. This paper questions whether transformers are really anything special for vision tasks, proposing that their success might just come from the usual approach of breaking images into patches.

The authors investigate this question by introducing a novel architecture called ConvMixer. In this approach, the input image is divided into patches using a standard convolution operation. This is followed by a series of blocks that incorporate pointwise convolutions, depthwise separable convolutions and residual connections.

One interesting thing about this approach is that it uses standard convolution to break down the image into patches, which is simpler (and more original) than what usual transformer methods propose. Also, the architecture is pretty simple and seems to performs well even with a limited number of trainable parameters. It's odd that there aren't any dropout layers, and we’ll explore that more later in the notebook.

# Implementation from scratch
You can have a look in the *convmixerlib* folder of this repository, which is a simple python custom module. All the code it's written using standard python type hints and docstrings, trying to keep things as simple as possible. Note that I intentionally minimized the number of functions to make the building blocks easier to understand and read (see note below).

Actually there is not much code at all, here's a short list of what you will find in each file of the *convmixerlib* module:
- **blocks.py** -> the convmixer basic building block
- **models.py** -> the convmixer network architecture
- **plots.py** -> simple function for plotting models training history (train and validation loss)
- **processing.py** -> functions for data augmentation and splitting data in training and validation sets

You can use alternatively the notebook "*convmixer-experiment-cifar10.ipynb*" or the script "*convmixer-experiment-cifar10.py*" to run an experiment on cifar-10 data using a convmixer network!

You can install the required dependencies using the "*requirements.txt*" file (WARNING -> jupyter notebooks dependencies not included), everything should work as expected using python 3.12 or similar versions.

**NOTE**: in this implementation i deliberately tried to use as few functions as possible. While it's useful to encapsulate repetetive basic building blocks (ex. convolution, batchnorm, activation) in functions, it quickly became a real nightmare when the only goal it's to understand a network design. When I look for network implementations and my goal is pure architectures understanding, I personally hate the functions inceptions when building simple stuff. Hope you will find easier to read the implementation in this way!
