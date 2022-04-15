#!/usr/bin/env python3

# TODO
"""Must downgrade numpy== 1.19.5
must pip install pillow

Source: https://www.kaggle.com/code/function9/bidirectional-lstm-gan-music-generation/notebook



To import a saved model

model = load_model('./LSTM_generator.h5')"""

from tensorflow.keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Bidirectional, LSTM, Reshape, RepeatVector,
# TimeDistributed

from tensorflow.keras.models import Sequential, Model
# from keras.optimizers import Adam
from numpy.random import rand, randint, randn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Input, Dense, Bidirectional, LSTM, \
    RepeatVector, TimeDistributed
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import re


run_gpu = input('Do you want to run on the GPU? : y/n    ')
if run_gpu == 'n':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class LSTMGAN:
    def __init__(self):
        self.dl = DataLoader()
        self.dl.loadData()
        self.data = self.dl.getData()

        # Input shape
        self.img_rows = self.dl.getMaxDimension()
        self.img_cols = self.dl.getMaxDimension()
        self.img_shape = (self.img_rows, self.img_cols)
        self.latent_dim = 16

        optimizer = Adam(0.0001, 0.4)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates song
        z = Input(shape=(self.dl.getMaxDimension(),self.dl.getMaxDimension()))

        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), return_sequences=True), input_shape=(self.dl.getMaxDimension(), self.dl.getMaxDimension())))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), return_sequences=True)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount())))
        model.add(LeakyReLU(alpha=0.2))

        #specifying output to have 40 timesteps
        model.add(RepeatVector(pow(self.dl.getMaxDimension(), 2)))
        #specifying 1 feature as the output
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), return_sequences=True, dropout = 0.2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), return_sequences=True, dropout = 0.2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), return_sequences=True, dropout = 0.2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.3))   
        model.add(TimeDistributed(Dense(self.dl.getUniqueWordCount())))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(self.dl.getUniqueWordCount())))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(1)))
        model.add(LeakyReLU(alpha=0.2))
        model.summary()


        noise = Input(shape=(self.dl.getMaxDimension(),self.dl.getMaxDimension()))

        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()


        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), activation = 'relu', return_sequences=True), input_shape=(pow(self.dl.getMaxDimension(), 2), 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Bidirectional(LSTM(self.dl.getUniqueWordCount(), activation = 'relu')))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(RepeatVector(1))
        model.add(TimeDistributed(Dense(self.dl.getUniqueWordCount(), activation = 'sigmoid')))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(self.dl.getUniqueWordCount(), activation = 'relu')))

        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(TimeDistributed(Dense(1, activation='linear')))
        model.summary()


        img = Input(shape=(pow(self.dl.getMaxDimension(), 2),1))

        validity = model(img)

        return Model(img, validity)
    
    def predict_and_save(self, epoch):
		
		# get the word dictionary from the data loader
        wordDict = self.dl.getSongWordDict(scaled=False)

        file = open('./savedSongs/generator_model_' + str(epoch) + '_songs.txt', 'w')
        for i in range(0, 10):
            random = np.random.normal(0, 1, (1, self.dl.getMaxDimension(), self.dl.getMaxDimension()))
            prediction = self.generator.predict(random)
            prediction = prediction * self.dl.getUniqueWordCount()

            prediction = prediction.reshape(self.dl.getMaxDimension()*self.dl.getMaxDimension(), 1)
            prediction = prediction.astype(int)

            #np.save('gan_prediction', prediction)
    
            song = []
            previousWord = None
            for word in prediction:
                # it is likely that the exact number predicted will not
                # match up to a value in the wordDict
                # so we will find the closest one that does
                wordDictValues = list(wordDict.values()) 
                        
                if not word == 0:
                    wordPredicted = list(wordDict.keys())[list(wordDict.values()).index(word)]
                    wordPredicted = wordPredicted.strip()
                    if not wordPredicted == '' and wordPredicted != previousWord:
                        song.append(wordPredicted)
                        previousWord = wordPredicted

            if not len(song) == 0 and not len(song) == 1:
                file.write(' '.join(song))
                file.write('\n--------------------\n')
        file.close()
    
    def train(self, epochs, batch_size, save_interval):

        # Load the dataset
        X_train = self.data

        # Rescale 0 to 1
        X_train = X_train / self.dl.getUniqueWordCount()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1, 1))
        fake = np.zeros((batch_size, 1, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of songs
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            imgs = np.array(imgs)

            imgs = imgs.reshape(len(imgs),pow(self.dl.getMaxDimension(), 2),1)

            # Sample noise and generate a batch of new songs
            noise = np.random.normal(0, 1, (batch_size,self.dl.getMaxDimension(),self.dl.getMaxDimension()))

            gen_imgs = self.generator.predict(noise)
                       
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake songs as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save model
            if epoch % save_interval == 0:
	        # generate a few songs with this current generator
                self.predict_and_save(epoch)
                self.generator.save(f"./saved-model-states/LSTM_generator_epoch_{epoch}.h5")


'''You can manually use DataLoader here instead of importing (which doesn't work all the time)'''
# class DataLoader:

#     # lowResAmount is the percentage that we
#     # would like to reduce the
#     def __init__(self, low_res_amount=.25):
#         self.data = []
#         self.allSongs = []
#         self.songWordDict = {}  # a dictionary of unique lyric words
#         self.songWordDictScaled = {}
#         self.uniqueWordCount = None
#         self.numpy_arrays = []
#         self.maxDimension = None
#         self.lowResAmount = low_res_amount

#     def loadData(self):
#         singleSong = []

#         # textLines = open('../data/baseline/genre_country_music.txt').readlines()
#         textLines = open('/home/kdl/Insync/OneDrive/School/Spring_2022/Machine_Learning/COMP-6630-Final-Project/data/baseline/genre_country_music.txt').readlines()


#         i = 1
#         for line in textLines:
#             words = line.split(' ')
#             if (words[0].strip() == 'TITLE:' or words[0].strip() == 'ARTIST:' or
#                     words[0].strip() == 'LYRICS:' or words[0].strip() == '/END' or
#                     words[0].strip() == '' or words[0].strip()[0] == '['):
#                 if words[0].strip() == '/END':
#                     self.allSongs.append(singleSong)
#                     singleSong = []
#                 continue
#             else:
#                 for word in words:
#                     w = word.lower().strip()
#                     w = re.sub(r'[^\w\s]', '', w)
#                     singleSong.append(w)
#                     if not w in self.songWordDict.keys():
#                         self.songWordDict.update({w: i})
#                         i += 1

#         self.uniqueWordCount = i - 1
#         self.createNumArrs()
#         self.createScaledWordDict()

#     def getData(self):
#         return self.data

#     def getUniqueWordCount(self):
#         return self.uniqueWordCount

#     def getMaxDimension(self):
#         return self.maxDimension

#     def getSongWordDict(self, scaled=False):
#         if not scaled:
#             return self.songWordDict
#         else:
#             return self.songWordDictScaled

#     def findMaxSongLength(self):
#         maxSongLength = 0
#         for song in self.allSongs:
#             if len(song) > maxSongLength:
#                 maxSongLength = len(song)

#         #print(maxSongLength)
#         return maxSongLength

#     def createScaledWordDict(self):
#         for k, v in self.songWordDict.items():
#             self.songWordDictScaled[k] = v / self.uniqueWordCount

#     def createNumArrs(self):

#         maxSongLength = self.findMaxSongLength()
#         self.maxDimension = self.getDimensions(maxSongLength)
#         maxArrLen = pow(self.maxDimension, 2)
#         # print('Max np array dimension: ', maxDimension, 'x', maxDimension, sep='')

#         for i in range(len(self.allSongs)):
#             songLength = len(self.allSongs[i])
#             songNArr = np.zeros(maxArrLen, dtype=int)
#             for h in range(songLength):
#                 word = self.allSongs[i][h]
#                 # get the unique identifier from the dict
#                 wordNum = self.songWordDict[word]
#                 # set this index in the numpy array to be that number
#                 songNArr[h] = wordNum
#             self.data.append(songNArr)

#         self.data = np.array(self.data)

#         self.data = self.data.reshape(self.data.shape[0], self.maxDimension, self.maxDimension)

#     def getDimensions(self, a, i=0):
#         if pow(i, 2) > a:
#             while not (i % (1 / self.lowResAmount) == 0):
#                 i += 1
#             #print(i)
#             return i
#         else:
#             return self.getDimensions(a, i + 1)
        
if __name__ == '__main__':
    dl = DataLoader()
    dl.loadData()
    #print(dl.getMaxDimension())
    #print(dl.getUniqueWordCount())
    lstmgan = LSTMGAN()
    lstmgan.train(epochs=20, batch_size=20, save_interval=2)

