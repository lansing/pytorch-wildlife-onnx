# Introduction

In this directory you will find the CameraTraps aka PytorchWildlife git repo. This is a set of models and utilities for wildlife detection and classification.

We are going to enhance this repo with our own scripts and utilities. We will include our enhancements in a directory tree outside of the CameraTraps repo (starting in the current directory).

## Task 1: Model Exporters

Create a utility to export the models into various formats. We will design this tool modularly, with modules for each model architecture we wish to support, as well as the formats we wish to export to.

In order to minimize code duplication, we will use these categories of components:


1. Model loader: loads the model we want to work with into memory (and access the pytorch module if the PytorchWildlife class is not one already). We need to be able to specify the model we want, i.e. megadetector v6 compact MIT. Let's create model loader classes for all the supported model architectures in the repo (consider if you can avoid repeating code by sharing one loader for multiple variants of the model, i.e. compact and large/extra would have one loader).

2. Model exporter: we should be able to create a base exporter class for each target export format, i.e. ONNX. We will create a concrete subclass for each model architecture. Initially we will assume that the subclass requires no overrides, i.e. the base class handles the general case. As we experiment we may discover that some additional logic is required for each model subtype.

2a. Model exporter CLI tool. Exports a selected model to the target format, using parts 1 and 2.

3. Model inspector/validator: a utility script to inspect the exported onnx model, metadata and nodes, for possible debugging use. Also validates that we can load the model graph.

4. Inference utilities: Repurpose/adapt any utils from pytorchwildlife, to use the exported onnx models for inference instead of the included pytorch models. look n the `demo` directory for example demos we want to repurpose.


Deliverable:
1. Model loaders for all supported model architectures, and unit tests for them

2. Model exporter for onnx format ONLY, with subclasses for each architecture. Intially we will validate the functionality using megadetector 6 yolov9 ultralytics, compact.

2a. model exporter must support different export numeric formats: 32 bit, 16 bit float, 8 bit int. Additionally, it must allow a 'simplify' step. And it must allow us to set onnx opcode, default will be 17.

3. Model inspector/validator. Check to ensure the exported model works and is what we expect. Test a forward pass with a randomly generated tensor to do preliminary valiation.


Order of operations:

0. Plan and create directory structure for the project.

1. Model inspector/validator. Develop against the onnx model in sample_models dir, MDV6-yolov9-c-320-16b.onnx. This is a known working, 16 bit yolov9 megadetector v6 model in onnx format. Assume that our exported models will be similar to this (not necessarily identical). Get the inspector/validator to a state where it can be used to later validate newly exported models using our new scripts.

2. Model loader: build model loaders for the models. Initially, just build them for two models: megadetector v6 yolov9 ultralytics compact, and megadetector v6 rtdetr apache compact. Unit tests for this.

3. model exporter: onnx exporter and subclasses for the supported models. Test the exporter using te validator/exporter.



