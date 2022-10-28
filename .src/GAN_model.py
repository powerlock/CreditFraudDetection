'''
Define the discriminator, that gets the image as input. 
Define a sequence of filters the model uses to classify this input image.
'''

from keras.layers import Input, Embedding, multiply, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers import ELU, PReLU, LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical
import numpy as np


class DataGan():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator(data_dim=29,num_classes=2)
        self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],loss_weights=[0.5, 0.5],optimizer=optimizer,metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator(latent_dim=10,data_dim=29)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self,latent_dim,data_dim):

            model = Sequential()

            model.add(Dense(16, input_dim=latent_dim))
        
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(32, input_dim=latent_dim))
        
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(data_dim,activation='tanh'))

            model.summary()

            noise = Input(shape=(latent_dim,))
            img = model(noise)

            return Model(noise, img)

    def build_discriminator(self, data_dim,num_classes):
        model = Sequential()
        model.add(Dense(31,input_dim=data_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.25))
        model.add(Dense(16,input_dim=data_dim))
        model.add(LeakyReLU(alpha=0.2))
        
        model.summary()
        img = Input(shape=(data_dim,))
        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(num_classes+1, activation="softmax")(features)
        return Model(img, [valid, label])

    def train(self,X_train,y_train,
            X_test,y_test,
            generator,discriminator,
            combined,
            num_classes,
            epochs, 
            batch_size=128):
        
        f1_progress = []
        half_batch = int(batch_size / 2)

        noise_until = epochs

        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        cw1 = {0: 1, 1: 1}
        cw2 = {i: num_classes / half_batch for i in range(num_classes)}
        cw2[num_classes] = 1 / half_batch

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 10))
            gen_imgs = generator.predict(noise)

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            labels = to_categorical(y_train[idx], num_classes=num_classes+1)
            fake_labels = to_categorical(np.full((half_batch, 1), num_classes), num_classes=num_classes+1)

            # Train the discriminator
            d_loss_real = discriminator.train_on_batch(imgs, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 10))
            validity = np.ones((batch_size, 1))

            # Train the generator
            g_loss = combined.train_on_batch(noise, validity, class_weight=[cw1, cw2])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))
            
            if epoch % 10 == 0:
                _,y_pred = discriminator.predict(X_test,batch_size=batch_size)
                #print(y_pred.shape)
                y_pred = np.argmax(y_pred[:,:-1],axis=1)
                
                f1 = f1_score(y_test,y_pred)
                print('Epoch: {}, F1: {:.5f}, F1P: {}'.format(epoch,f1,len(f1_progress)))
                f1_progress.append(f1)
                
        return f1_progress
if __name__== "__main__":
    generator = DataGan.build_generator(latent_dim=10,data_dim=29)
    discriminator = DataGan.build_discriminator(data_dim=29,num_classes=2)
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
    loss_weights=[0.5, 0.5],
    optimizer=optimizer,
    metrics=['accuracy'])

    noise = Input(shape=(10,))
    img = DataGan.generator(noise)
    discriminator.trainable = False
    valid,_ = discriminator(img)
    combined = Model(noise , valid)
    combined.compile(loss=['binary_crossentropy'],
        optimizer=optimizer)


    f1_p = DataGan.train(X_res,y_res,
             X_test,y_test,
             generator,discriminator,
             combined,
             num_classes=2,
             epochs=5000, 
             batch_size=128)