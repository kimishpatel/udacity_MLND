# udacity_MLND

# Project title
Capstone project: Street view house number recognition. Written in python using tensorflow library

## Prerequisites

Required python libraries are: sklearn, numpy, scipy, tensorflow, hdf5

## Prerun steps

Following steps will extract training/test images, resize them and form a pickle file containing traing and test data:

```
cd capstone/utils
```
In config.ini file you can specifiy the dimensions of the resized image. Default is 50x50.
```
python download_and_extract.py
```
After this step you should see a data folder which will contain a subfolder 50x50 (of whatever dimensions you configured in config.ini). This folder will contain training and test images.

```
python convert_to_grayscale.py
```
This step will convert all images to grayscale and resize them to the dimensions specified in config.ini (50x50 by default). Original images will be DELETED.

```
python generate_train_test_data.py
```
This step generates two pickle files. One containing original train and test the other one formed after mixing training and test data, shuffling it and repartitioning it.

## Training neural network on input data

There are two approaches I undertook to implement CNN based classifier. One that predicts house number length and digits out of single network called Unified (performs beter). The other employs separate network, called Distributed, for the tasks for length prediction and individual digit prediction (5 of those, 1 for length 4 for digits, I ignored 5th one). 
Unified can be run as follows:
```
python SVHN_train_and_classify_entire_number.py -f svhn_network_entire_number.json
```
For distributed you can simply use run_script.sh and run that which will kick off each of the training steps one after the other

