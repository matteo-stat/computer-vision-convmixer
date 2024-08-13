# Introduction
I believe that one of the most fascinating aspects of deep learning is the development of network architectures for computer vision applications. Since 2012, many architectures have been proposed. While some building blocks have become standard, the overall structure of these networks often looks drastically different from one another.

Some architectures appear simple, others complex, and some, let's say, simply original! These differences highlight the creative process often involved in developing network architectures for computer vision applications. Thanks to modern GPUs available in consumer computers, it's now quite feasible to design your own architecture, experiment with ideas, and start training from scratch.

# ConvMixer Network - "Patches Are All You Need?"
Recently, I came across the paper ["Patches are all you need?"](https://arxiv.org/abs/2201.09792) and found it to be quite original. In the past few years, transformer architectures have gained popularity in computer vision, often achieving excellent results but demanding more resources than traditional CNNs. This paper questions whether transformers are really anything special for vision tasks, proposing that their success might just come from the usual approach of breaking images into patches.

The authors investigate this question by introducing a novel architecture called ConvMixer. In this approach, the input image is divided into patches using a standard convolution operation. This is followed by a series of blocks that incorporate pointwise convolutions, depthwise separable convolutions and residual connections.

One interesting thing about this approach is that it uses standard convolution to break down the image into patches, which is simpler (and more original) than what usual transformer methods propose. Also, the architecture is pretty simple and seems performs well even with a limited number of trainable parameters. It's odd that there aren't any dropout layers, and we’ll explore that more later in the notebook.

# Implementation from scratch
You can have a look in the convmixerlib folder of this repository, which is a simple python custom module. All the code it's written using typing hints and docstrings, trying to keep things as simple as possible.

Actually there is not much code at all, here's a short list of what you will find in each file of the convmixerlib module:
- blocks -> the convmixer basic building block
- models -> the convmixer network architecture
- plots -> simple function for plotting models training history (train and validation loss)
- processing -> functions for data augmentation and splitting data in training and validation sets

You can use alternatively the notebook "convmixer-experiment-cifar10.ipynb" or the script "convmixer-experiment-cifar10.py" to run an experiment on cifar-10 data using a convmixer network!
