# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from numpy import expand_dims, ones, zeros, vstack, full
from numpy.random import randint, randn

from dataloader import DataLoader


class GAN():

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.low_res_amount = .25
        self.discrimConvCount = 2
        self.dl = DataLoader(low_res_amount=self.low_res_amount)

    def load_real_samples(self):
        # load the data
        self.dl.loadData()
        data = self.dl.getData()

        print('Data Shape: ', data.shape)

        # expand the data to 3d, e.g. add channels dimension
        X = expand_dims(data, axis=-1)
        # convert from unsigned ints to floats
        X = X.astype('float32')

        # scale from 0 to the uniqueWordCount of the dataset to 0, 1
        X = X / self.dl.getUniqueWordCount()

        return X

    # generate n fake samples with class labels
    def generate_fake_samples(self, g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n_samples)
        # predict ouputs
        X = g_model.predict(x_input)
        # generate 'fake' class labels (0)
        y = zeros((n_samples, 1))

        return X, y

    # select real samples
    def generate_real_samples(self, dataset, n_samples):
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)

        # retrieve selected images
        X = dataset[ix]

        # generate 'real' class labels (0.9)
        # we will use smooth sampling here and choose 0.9
        # instead of 1 for the output labels

        y = full((n_samples, 1), 0.9)

        return X, y

    # define the standalone discriminator model
    def define_discriminator(self, in_shape):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def train_discriminator(self, model, dataset, n_iter=100, n_batch=2):
        half_batch = int(n_batch / 2)

        # manually enumerate epochs
        for i in range(n_iter):
            # get randomly selected 'real' samples
            X_real, y_real = self.generate_real_samples(dataset, half_batch)

            # update discriminator on real samples
            _, real_acc = model.train_on_batch(X_real, y_real)

            # generate 'fake' examples
            X_fake, y_fake = self.generate_fake_samples(half_batch)

            # update discriminator on fake samples
            _, fake_acc = model.train_on_batch(X_fake, y_fake)

            # summarize performance
            print('>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))

    def define_generator(self, maxDim, latentDim):
        model = Sequential()
        # foundation for a low resolution version of our image
        # this will be n(m/2) where m is our maxDimension from our dataloader
        # and n is the number of Conv2D layers that we have in our discriminator
        lowResLayerVal = maxDim / (self.discrimConvCount * 2)

        n_nodes = 128 * int(lowResLayerVal) * int(lowResLayerVal)
        model.add(Dense(n_nodes, input_dim=latentDim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((int(lowResLayerVal), int(lowResLayerVal), 128)))
        # upsample to lowResLayerVal * 2 by lowResLayerVal * 2
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample again to lowResLayerVal * 4 by lowResLayerVal * 4
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))

        return model

    def generate_latent_points(self, latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add the discriminator
        model.add(d_model)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train_gan(self, gan_model, latent_dim, n_epochs=100, n_batch=8):
        # manually enumerate epochs
        for i in range(n_epochs):
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_can = ones((n_batch, 1))
            # update the generator via the discriminator's error
            gan_model.tran_on_batch(x_gan, y_gan)

    # train the generator and discriminator
    def train(self, g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=8):
        bat_per_epo = int(dataset.shape[0] / n_batch)
        half_batch = int(n_batch / 2)

        # manually enumerate epochs
        for i in range(n_epochs):
            # enumarate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real'samples
                X_real, y_real = self.generate_real_samples(dataset, half_batch)
                # generate 'fake' exmaples
                X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
                # create training set for the discriminator
                X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))

                # update discrminiator model weights
                d_loss, _ = d_model.train_on_batch(X, y)
                # prepare points in latent space as input for the generator
                X_gan = self.generate_latent_points(latent_dim, n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discrminator's error
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch

            # evaluate the model performance
            if (i + 1) % 10 == 0:
                self.summarize_performance(i, g_model, d_model, dataset, latent_dim)
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch, g_model, d_model, dataset, latent_dim, n_samples=10):
        # prepare real samples
        X_real, y_real = self.generate_real_samples(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, n_samples)
        # evaluate discrminiator on fake examples
        _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
        # summarize discrimiator performace
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
        # save songs
        self.save_songs(X_fake, epoch)
        # save the generator model tile file
        filename = 'saved-model-states/generator_model_%03d.h5' % (epoch + 1)
        g_model.save(filename)

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

    def run(self):
        dataset = self.load_real_samples()

        d_Model = self.define_discriminator((self.dl.getMaxDimension(), self.dl.getMaxDimension(), 1))

        d_Model.summary()
        g_Model = self.define_generator(self.dl.getMaxDimension(), self.latent_dim)

        g_Model.summary()

        gan_model = self.define_gan(g_Model, d_Model)

        gan_model.summary()

        self.train(g_Model, d_Model, gan_model, dataset, self.latent_dim)


if __name__ == "__main__":
    gan = GAN(latent_dim=250)
    gan.run()
