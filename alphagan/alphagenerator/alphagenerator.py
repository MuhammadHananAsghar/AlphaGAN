import tensorflow as tf
import os
import numpy as np

class AlphaDatagen(tf.keras.utils.Sequence):
    def __init__(self, paths, SEED_PATH,
                 batch_size,
                 input_size=(128, 128, 3)):
        self.df = paths.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.PATH = SEED_PATH
        
        self.n = len(self.df)
        self.m = 0
        self.max = self.__len__()
    
    def __len__(self):
        return int(len(self.df) / float(self.batch_size))
    
    def __get_input(self, path, target_size):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        image_arr = image_arr / 127.5 - 1.
        return image_arr
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in batches])
        return X_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X_batches = [os.path.join(self.PATH, img) for img in batches]
        X = self.__get_data(X_batches)
        return X
    
    def __next__(self):
        if self.m >= self.max:
            self.m = 0
        Y = self.__getitem__(self.m)
        self.m += 1
        return Y