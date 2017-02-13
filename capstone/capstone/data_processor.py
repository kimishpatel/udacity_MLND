from sklearn import cross_validation as cv
from sklearn.utils import shuffle 
from six.moves import cPickle as pickle
from scipy import ndimage
import numpy as np

class DataProcessor:
    def __init__(self):
        self.message = "This is an empty class"

    @staticmethod
    def load_data(pickle_file, extra_data=False):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            if not extra_data:
                train_dataset = data['input_data']
                train_labels = data['input_labels']
                test_dataset = data['test_data']
                test_labels = data['test_labels']
                del data  # hint to help gc free up memory
                print('Training set', train_dataset.shape, train_labels.shape)
                print('Test set', test_dataset.shape, test_labels.shape)
                return train_dataset, train_labels, test_dataset, test_labels
            else:
                test_dataset = data['extra_data']
                test_labels = data['extra_labels']
                print('Test set', test_dataset.shape, test_labels.shape)
                return None, None, test_dataset, test_labels

    @staticmethod
    def load_and_split_data(pickle_file, stratify=True, label_index=None):
        tr_size = 0.8
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            train_dataset = data['input_data']
            if label_index != None:
                train_labels = data['input_labels'][:,label_index]
            else:
                train_labels = data['input_labels']
            print('train dataset size', train_dataset.shape)
            print('train labels size', train_labels.shape)
            test_dataset = data['test_data']
            if label_index != None:
                test_labels = data['test_labels'][:,label_index]
            else:
                test_labels = data['test_labels']
            del data  # hint to help gc free up memory
            #do 80 20 split of train into train and valid
            if stratify:
                train_dataset, valid_dataset, train_labels, valid_labels = cv.train_test_split(train_dataset, train_labels, train_size = tr_size, stratify = train_labels,random_state=40)
            else:
                train_dataset, valid_dataset, train_labels, valid_labels = cv.train_test_split(train_dataset, train_labels, train_size = tr_size, random_state=40)
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
     
    @staticmethod
    def normalize_data(datasets, mean_val, val_range):
        output_dataset = list()
        for data in datasets:
            data = (data - float(mean_val)) / float(val_range)
            output_dataset.append(data)
        return output_dataset
     
    @staticmethod
    def normalize(data, mean_val, val_range):
        data = (data - float(mean_val)) / float(val_range)
        return data
     
    @staticmethod
    def denormalize(data, mean_val, val_range):
        data = (data*val_range) + mean_val
        return data
     
    @staticmethod
    def apply_edge_filter(image_list, alpha = 8):
        print('Edge extraction')
        for index in range(image_list.shape[0]):
            image = image_list[index, :, :]
            im = ndimage.gaussian_filter(image, alpha)
            image_list[index,:,:] = image - im 
        return image_list

    @staticmethod
    def dump_data(data_dict, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL) 

    @staticmethod
    def chop_data(fraction, input_data, labels):
        size = input_data.shape[0]
        indices = np.random.choice(range(size), int(size*fraction))
        input_data = input_data[indices,...]
        labels = labels[indices,...]
        return input_data, labels

    #use this method to filter data where -ve examples outweight +ve
    #here we will not learn much otherwise
    @staticmethod
    def filter_data(input_data, input_labels):
        filtered_indices = np.where(input_labels != -1)
        input_data_positives = input_data[filtered_indices[0],...]
        input_labels_positives = input_labels[filtered_indices[0]]
        print('Filtered dataset', input_data_positives.shape, input_labels_positives.shape)
        filtered_indices_negatives = np.where(input_labels == -1)
        filtered_indices_negatives = np.random.choice(filtered_indices_negatives[0], len(filtered_indices[0]))
        input_data_negatives = input_data[filtered_indices_negatives,...]
        input_labels_negatives = input_labels[filtered_indices_negatives]

        output_data = np.ndarray(shape=(input_data_positives.shape[0]+input_data_negatives.shape[0], input_data_positives.shape[1], input_data_positives.shape[2]))
        output_labels = np.ndarray(shape=(input_data_positives.shape[0]+input_data_negatives.shape[0], ))
        output_data[0:input_data_positives.shape[0],...] = input_data_positives
        output_labels[0:input_data_positives.shape[0]] = input_labels_positives
        output_data[input_data_positives.shape[0]:,...] = input_data_negatives
        output_labels[input_data_positives.shape[0]:] = input_labels_negatives

        #now shuffle otherwise positive examples are at the begninning and -ve at the end
        output_data, output_labels = shuffle(output_data, output_labels)

        print('Filtered dataset with negatives', output_data.shape, output_labels.shape)
        return output_data, output_labels 

