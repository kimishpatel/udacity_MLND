from __future__ import print_function
import os
import numpy as np
import random
import string
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tarfile
import configparser

url = 'http://ufldl.stanford.edu/housenumbers/'
to_location = '../data/'

def maybe_download_and_extract(filename):
  if not os.path.isdir(to_location):
      os.makedirs(to_location)
  tarfile_name = filename+'.tar.gz' 
  download_to = to_location + filename + '.tar.gz'
  if not os.path.exists(download_to):
      print("downloading file ", url+tarfile_name, "to ", download_to)
      try:
          _, _ = urlretrieve(url + tarfile_name, download_to)
      except IOError, e:
          print("Can't retrieve %r to %r: %s" % (theurl, thedir, e))
          exit(-1)
  print("extracting file ", download_to)
  try:
      tar = tarfile.open(download_to)
  except Exception as e:
      print(str(e))
      exit(-1)
  tar.extractall(to_location)
  tar.close()

try:
    config = configparser.ConfigParser()
    config.read('../config.ini')
    image_width = config['default']['image_width']
    image_height = config['default']['image_height']
    to_location = '../data/'+str(image_width)+'x'+str(image_height)+'/'
except Exception as e:
    print(str(e))
    to_location = '../data/'

maybe_download_and_extract('train')
maybe_download_and_extract('test')

