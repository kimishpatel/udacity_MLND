#!/usr/bin/python

# Ref:https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python 
#Kimish's note: I took this code for reading digitStruct mat file from SVHN database
#Thanks to github user prijip for this.

import h5py
import numpy as np

#
# Bounding Box
#
class HouseDigits:
    def __init__(self):
        self.digits= [] #list of digits
        self.num_digits = 0 #how many digits are there in this house number

class DigitStruct:
    def __init__(self):
        self.name = None    # Image file name
        self.houseDigits = None # List of BBox structs

# Function for debugging
def printHDFObj(theObj, theObjName):
    isFile = isinstance(theObj, h5py.File)
    isGroup = isinstance(theObj, h5py.Group)
    isDataSet = isinstance(theObj, h5py.Dataset)
    isReference = isinstance(theObj, h5py.Reference)
    print "{}".format(theObjName)
    print "    type(): {}".format(type(theObj))
    if isFile or isGroup or isDataSet:
        # if theObj.name != None:
        #    print "    name: {}".format(theObj.name)
        print "    id: {}".format(theObj.id)
    if isFile or isGroup:
        print "    keys: {}".format(theObj.keys())
    if not isReference:
        print "    Len: {}".format(len(theObj))

    if not (isFile or isGroup or isDataSet or isReference):
        print theObj

def readDigitStructGroup(dsFile):
    dsGroup = dsFile["digitStruct"]
    return dsGroup

#
# Reads a string from the file using its reference
#
def readString(strRef, dsFile):
    strObj = dsFile[strRef]
    str = ''.join(chr(i) for i in strObj)
    return str

#
# Reads an integer value from the file
#
def readInt(intArray, dsFile):
    intRef = intArray[0]
    isReference = isinstance(intRef, h5py.Reference)
    intVal = 0
    if isReference:
        intObj = dsFile[intRef]
        intVal = int(intObj[0])
    else: # Assuming value type
        intVal = int(intRef)
    if intVal == 10:
        intVal = 0
    return intVal

def yieldNextInt(intDataset, dsFile):
    for intData in intDataset:
        intVal = readInt(intData, dsFile)
        yield intVal 

def yieldNextBBox(bboxDataset, dsFile):
    for bboxArray in bboxDataset:
        bboxGroupRef = bboxArray[0]
        bboxGroup = dsFile[bboxGroupRef]
        labelDataset = bboxGroup["label"]

        houseDigits = HouseDigits()
        num_digits = 0

        for label in yieldNextInt(labelDataset, dsFile):
            houseDigits.digits.append(label)
            num_digits += 1

        houseDigits.num_digits = num_digits
        yield houseDigits

def yieldNextFileName(nameDataset, dsFile):
    for nameArray in nameDataset:
        nameRef = nameArray[0]
        name = readString(nameRef, dsFile)
        yield name

# dsFile = h5py.File('../data/gsvhn/train/digitStruct.mat', 'r')
def yieldNextDigitStruct(dsFileName):
    dsFile = h5py.File(dsFileName, 'r')
    dsGroup = readDigitStructGroup(dsFile)
    nameDataset = dsGroup["name"]
    bboxDataset = dsGroup["bbox"]

    digitsIter = yieldNextBBox(bboxDataset, dsFile)
    for name in yieldNextFileName(nameDataset, dsFile):
        digitsList = next(digitsIter)
        obj = DigitStruct()
        obj.name = name
        obj.houseDigits = digitsList
        yield obj

def ParseDigitStruct(filename, debug=False):
    dsFileName = filename
    too_long_house_numbers = 0
    houseList = []
    for dsObj in yieldNextDigitStruct(dsFileName):
        if debug:
            print " name:{} num digits:{}".format(dsObj.name, dsObj.houseDigits.num_digits)
            for digit in dsObj.houseDigits.digits:
                print(" "+str(digit))
        if dsObj.houseDigits.num_digits > 5:
            print("Found a house number that is too long")
            too_long_house_numbers += 1
        else:
            houseList.append(dsObj)

    return houseList

if __name__ == "__main__":
    testMain('../data/train/digitStruct.mat')

