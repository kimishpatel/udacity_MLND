from sklearn import cross_validation as cv
from six.moves import cPickle as pickle
from PIL import Image
import numpy as np

class Photo:
    def __init__(self, filename):
        self.image_name = filename
        self.image = Image.open(filename)
        self.width = self.image.size[0]
        self.height = self.image.size[1]
        print "processed file: {} size {}".format(self.image_name,self.image.size)

    def convert2grayscale(self):
        self.image = self.image.convert('L')

    def convert2nparray(self):
        print "{} {}".format(self.width, self.height)
        return np.array(self.image.getdata(), np.uint8).reshape(self.width, self.height)
        #return np.array(self.image)

    def save(self,suffix):
        ext_idx = self.image_name.rfind('.')
        self.image_name = self.image_name[:ext_idx]+suffix+self.image_name[ext_idx:]
        self.image.save(self.image_name)

    def resize(self, new_size):
        #new_size is a tuple of (new_width, new_height)
        self.image = self.image.resize(new_size, Image.ANTIALIAS)

