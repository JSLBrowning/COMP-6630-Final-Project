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
    
    def predict(self):
        random = np.random.normal(0, 1, (1, self.dl.getMaxDimension(), self.dl.getMaxDimension()))

        prediction = self.generator.predict(random)

        prediction = prediction * self.dl.getUniqueWordCount()

        print(prediction)
    
    
    # create and save a file generated songs
    def save_songs(self, examples, epoch, n=10):

        # get the word dictionary from the data loader
        wordDict = self.dl.getSongWordDict(scaled=False)

        file = open('./savedSongs/' + 'generator_model_' + str(epoch + 1) + '_songs.txt', 'w')
        for i in range(len(examples)):
            try:
                file.write('\nSong #{count}\n'.format(count=i + 1))
                for line in examples[i]:
                    songLine = []
                    for word in line:

                        # it is likely that the exact number predicted will not
                        # match up to a value in the wordDict
                        # so we will find the closest one that does
                        wordDictValues = list(wordDict.values())
                        
                        # scale the word predicted back up to normal levels
                        word = round(word[0] * self.dl.getUniqueWordCount())
                        
                        
                        if not word == 0:
                            print('c')
                            wordPredicted = list(wordDict.keys())[list(wordDict.values()).index(word)]
                            wordPredicted = wordPredicted.strip()
                            if not wordPredicted == '':
                                songLine.append(wordPredicted)

                    if not len(songLine) == 0 and not len(songLine) == 1:
                        file.write(' '.join(songLine) + '\n')
                file.write('\n')
            except Exception as e:
                print("Unable to write to song file")
                print(e)

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
            
            #Save generator output songs
            self.save_songs(examples = gen_imgs,epoch = epoch, n=10)
            

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
                self.generator.save(f"./saved-model-states/LSTM_generator_epoch_{epoch}.h5")


if __name__ == '__main__':

    dl = DataLoader()
    dl.loadData()
    print(dl.getMaxDimension())
    print(dl.getUniqueWordCount())
    lstmgan = LSTMGAN()
    lstmgan.train(epochs=20, batch_size=20, save_interval=2)
    lstmgan.predict()
