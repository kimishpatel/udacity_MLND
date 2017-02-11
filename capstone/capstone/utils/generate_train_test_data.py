from six.moves import cPickle as pickle
from PIL import Image
import numpy as np
import digitStruct
from image_processor import *
import sys
sys.path.append('../')
from data_processor import *
import gc
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import configparser

#image size
width, height = 50, 50
house_num_max_length = 5
suffix = '__G_resized.png'
base_path = '../data/'
pickle_file = '../data/50x50/SVHN_data.pickle'
data_path = '../data/50x50/'

class SVHN_Utils: 

    @staticmethod
    def mix_train_test_data(input_file, output_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
            train_dataset = data['input_data']
            train_labels = data['input_labels']
            test_dataset = data['test_data']
        test_labels = data['test_labels']
    merged_data = np.ndarray(shape=(train_dataset.shape[0]+test_dataset.shape[0], train_dataset.shape[1], train_dataset.shape[2]))
    merged_labels = np.ndarray(shape=(train_dataset.shape[0]+test_dataset.shape[0], train_labels.shape[1]))
    merged_data[:train_dataset.shape[0],] = train_dataset
    merged_data[train_dataset.shape[0]:,] = test_dataset
    merged_labels[:train_dataset.shape[0],] = train_labels
    merged_labels[train_dataset.shape[0]:,] = test_labels

    random_indices = np.random.permutation(merged_data.shape[0])
    train_dataset = merged_data[random_indices[0:train_dataset.shape[0]],...]
    test_dataset = merged_data[random_indices[train_dataset.shape[0]:],...]
    train_labels = merged_labels[random_indices[0:train_dataset.shape[0]],]
    test_labels = merged_labels[random_indices[train_dataset.shape[0]:],]
    print('train dataset size', train_dataset.shape)
    print('train labels size', train_labels.shape)
    train_test_data = {'input_data': train_dataset, 'input_labels': train_labels, 'test_data': test_dataset, 'test_labels': test_labels}
    DataProcessor.dump_data(train_test_data, output_file)

    @staticmethod
    def plot_images(data, labels, num_indices, width, height, image_name):
        fig = plt.figure()
        indices = np.random.choice(range(num_indices), 4)
        for i in range(4):
            a = fig.add_subplot(2, 2, i+1)
            image = data[indices[i]]
            image = image.reshape((height, width))
            print('shape ',image.shape)
        print(image)
            image = image.squeeze()
            implot = plt.imshow(image, cmap = 'Greys_r')
            a.set_title((labels[indices[i]]))
        if image_name == None:
            plt.show()
        else:
        plt.savefig(image_name)

    @staticmethod
    def GenerateBatch(dsFile, image_dir, image_name, size=None):
        houseList = digitStruct.ParseDigitStruct(dsFile)
        num_houses = len(houseList)
        if size != None:
            num_houses = size
        #generate numpy array for images
        images = np.ndarray((num_houses, width, height), dtype=np.int16)
        #first label is the length of the house number in digits
        #rest are the actual digits.
        labels = np.ndarray((num_houses, 1+house_num_max_length), dtype=np.int8)
        #fill the array with -1. Thus missing digits are assumed to be -1
        labels.fill(-1)
        for house,i in zip(houseList, range(num_houses)):
            name = house.name
            houseDigits = house.houseDigits
            #split name because it already contains extension
            image = Photo(image_dir+"/"+name.split('.')[0]+suffix)
            npimage = image.convert2nparray()
            #add image to the database
            images[i,...] = npimage
            labels[i,0] = houseDigits.num_digits
            for (digit,j) in zip(houseDigits.digits, range(houseDigits.num_digits)):
                labels[i,j+1] = digit
        houseList = None
        #SVHN_Utils.plot_images(images, labels, num_houses, width, height, image_name)
        gc.collect()
        return images, labels

    @staticmethod
    def plot_histogram(data, outfile, num_bins = 5, label_bars=False):
        histo, bins = np.histogram(data, bins = num_bins)
        fig, ax = plt.subplots()
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        bottom = (0.,)*(len(bins)-1)
        print(histo.shape)
        print(center.shape)
        print(center)
        print(bottom)
        #_, _, rects = ax.bar(center, histo, align='center', width=width)
        plt.bar(center, histo, align='center', width=width)
        plt_title = 'histogram mean:'+str(round(np.mean(data), 2))+' std:'+str(round(np.std(data), 2))+' median:'+str(round(np.median(data), 2))
        plt.title(plt_title)
        plt.savefig(outfile,bbox_inches='tight')
        plt.show()

    @staticmethod
    def DigitsDistribution(dsFile, outfile):
        houseList = digitStruct.ParseDigitStruct(dsFile)
        num_houses = len(houseList)
        house_num_digits = []
        for house in houseList:
            house_num_digits.append(house.houseDigits.num_digits)
        SVHN_Utils.plot_histogram(house_num_digits, outfile)

    @staticmethod
    def DataDistribution(pickle_file, outfile, datakey):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            train_dataset = data[datakey]
            del data  # hint to help gc free up memory
            print('Training set', train_dataset.shape)
            SVHN_Utils.plot_histogram(train_dataset.flatten(), outfile, num_bins = 50)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    try:
    config.read('../config.ini')
    width = int(config['default']['image_width'])
    height = int(config['default']['image_height'])
    except Exception as e:
    print("could not read config file because ", str(e))
    data_path = base_path+str(width)+'x'+str(height)+'/' 
    pickle_file = data_path+'SVHN_data.pickle' 
       
    train_images, train_labels = SVHN_Utils.GenerateBatch(data_path+'train/digitStruct.mat',data_path+'train', 'train.png')
    test_images, test_labels = SVHN_Utils.GenerateBatch(data_path+'test/digitStruct.mat',data_path+'test', 'test.png')
    train_test_data = {'input_data': train_images, 'input_labels': train_labels, 'test_data': test_images, 'test_labels': test_labels}
    DataProcessor.dump_data(train_test_data, pickle_file)

    #in case you want to plot the distribution of data
    #SVHN_Utils.DigitsDistribution(data_path+'train/digitStruct.mat','training_num_digits_histo.png')
    #SVHN_Utils.DataDistribution(pickle_file,'training_data_histo.png', 'input_data')
    #SVHN_Utils.DataDistribution(pickle_file,'test_data_histo.png', 'test_data')
    SVHN_Utils.mix_train_test_data(pickle_file, data_path+'SVHN_data_shuffled.pickle')

    ##getting some extra data if available, must be downloaded first
    #extra_images, extra_labels = SVHN_Utils.GenerateBatch(data_path+'extra/digitStruct.mat',data_path+'extra', 'extra.png', 20000)
    #extra_test_data = {'extra_data': extra_images, 'extra_labels': extra_labels}
    #DataProcessor.dump_data(extra_test_data, data_path+'extra_svhn_data.pickle')
    #SVHN_Utils.DigitsDistribution(data_path+'extra/digitStruct.mat','extra_num_digits_histo.png')
    #SVHN_Utils.DataDistribution(data_path+'extra_svhn_data.pickle','extra_data_histo.png', 'extra_data')

