import glob
from image_processor import *
import os
import matplotlib.pyplot as plt
import numpy as np
import configparser

height_data = []
width_data = []
#path = '../data/train/*.png'
path = '../data/50x50/train/*.png'
image_width = 50
image_height = 50

def convert2grayscale(path):
    for image in glob.glob(path):
        print("processing "+image)
        image_data = Photo(image)
        #image_data.save('_G')
        height_data.append(image_data.height)
        width_data.append(image_data.width)
        #os.remove(image)
    return (height_data, width_data)

#resize all the images specified in the path
def convert_and_resize_all(path, convert=False):
    for image in glob.glob(path):
        print("processing "+image)
        image_data = Photo(image)
        if convert:
            image_data.convert2grayscale()
        image_data.resize((image_width, image_height))
        image_data.save('__G_resized')
        os.remove(image)

def plot_histogram(height_data, width_data):
    #plot histogram
    #print(sorted(height_data))
    plt.hist(height_data, bins = 50)
    plt_title = 'height histogram min:'+str(min(height_data))+' max:'+str(max(height_data))+' median:'+str(np.median(height_data))
    plt.title(plt_title)
    plt.savefig('height_histo.png',bbox_inches='tight')
    plt.show()

    #print(sorted(width_data))
    plt.hist(width_data, bins=50)
    plt_title = 'width histogram '+str(min(width_data))+' max:'+str(max(width_data))+' median:'+str(np.median(width_data))
    plt.title(plt_title)
    plt.savefig('width_histo.png',bbox_inches='tight')
    plt.show()

config = configparser.ConfigParser()
try:
    config.read('../config.ini')
    image_width = int(config['default']['image_width'])
    image_height = int(config['default']['image_height'])
except Exception as e:
    print("could not read config file because ", str(e))
   
base_path = '../data/'+str(image_width)+'x'+str(image_height)
#convert and resize train data
path = base_path+'/train/*.png' 
#(height_data, width_data) = convert2grayscale(path)
#plot_histogram(height_data, width_data)
convert_and_resize_all(path, convert=True)
path = base_path+'/test/*.png' 
convert_and_resize_all(path, convert=True)
#path = base_path+'/extra/*.png'
#convert_and_resize_all(path, convert=True)
