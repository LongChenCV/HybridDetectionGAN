from keras.layers import Lambda, Layer, Input, Conv2D, Activation, Multiply, multiply,Add, Concatenate, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from copy import deepcopy
from bs4 import BeautifulSoup
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder

from collections import OrderedDict
from scipy.misc import imsave, toimage
import numpy as np
import random
import time
import json
import csv
import sys
import os

import keras.backend as K
import tensorflow as tf

import load_data as load_data
np.random.seed(seed=12345)
class alphaRLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(alphaRLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alphaR = self.add_weight(name='alphaR',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(alphaRLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.exp(tf.multiply(-1.0, tf.multiply(self.alphaR, x)))

class alphaGLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(alphaGLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alphaG = self.add_weight(name='alphaG',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(alphaGLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.exp(tf.multiply(-1.0, tf.multiply(self.alphaG, x)))

class alphaBLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(alphaBLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.alphaB = self.add_weight(name='alphaB',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(alphaBLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.exp(tf.multiply(-1.0, tf.multiply(self.alphaB, x)))

class bglightLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(bglightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bglight = self.add_weight(name='bglight',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(bglightLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.multiply(self.bglight, x)

class bsbetaRLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(bsbetaRLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bsbetaR = self.add_weight(name='bsbetaR',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(bsbetaRLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.subtract(1.0, tf.exp(tf.multiply(-1.0, tf.multiply(self.bsbetaR, x))))

class bsbetaGLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(bsbetaGLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bsbetaR = self.add_weight(name='bsbetaG',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(bsbetaGLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.subtract(1.0, tf.exp(tf.multiply(-1.0, tf.multiply(self.bsbetaG, x))))

class bsbetaBLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(bsbetaBLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bsbetaR = self.add_weight(name='bsbetaB',
                                       shape=(1,),
                                       initializer='uniform',
                                       trainable=True)
        super(bsbetaBLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.subtract(1.0, tf.exp(tf.multiply(-1.0, tf.multiply(self.bsbetaB, x))))

class CycleGAN():
    def __init__(self, lr_D=2e-4, lr_G=2e-4, image_shape=(512, 512, 3),
                 date_time_string_addition='_test', image_folder=''):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = InstanceNormalization
        self.ofperceptor = True
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_depth = 1.0  # Cyclic loss depth
        self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.lambda_P = 1.0  # Weight for loss from preceptor on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 5  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 200  # choose multiples of 25 since the models are save each 25th epoch
        self.save_interval = 20
        self.synthetic_pool_size = 50
        self.n_classes = 5
        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = False
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start
        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number
        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True
        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = True
        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss
        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        if self.ofperceptor:
            D_A = self.modelDiscriminatorA_OF()
        else:
            D_A = self.modelDiscriminatorA()
        D_B = self.modelDiscriminatorB()
        loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images

        # Discriminator builds
        if self.ofperceptor:
            image_A = Input(shape=self.img_shape)
            image_B = Input(shape=self.img_shape)
            guess_A = D_A(image_A)
            guess_B = D_B(image_B)
            self.D_A = Model(inputs=image_A, outputs=guess_A, name='D_A_model')
            self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')
            self.D_A.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
            self.D_B.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
            self.D_A_static = Network(inputs=image_A, outputs=guess_A, name='D_A_static_model')
            self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')
        else:
            image_A = Input(shape=self.img_shape)
            depth_A = Input(shape=(512, 512, 1))
            image_B = Input(shape=self.img_shape)
            guess_A = D_A([image_A, depth_A])
            guess_B = D_B(image_B)
            self.D_A = Model(inputs=[image_A, depth_A], outputs=guess_A, name='D_A_model')
            self.D_B = Model(inputs=image_B, outputs=guess_B, name='D_B_model')
            self.D_A.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
            self.D_B.compile(optimizer=self.opt_D, loss=self.lse, loss_weights=loss_weights_D)
            self.D_A_static = Network(inputs=[image_A, depth_A], outputs=guess_A, name='D_A_static_model')
            self.D_B_static = Network(inputs=image_B, outputs=guess_B, name='D_B_static_model')

        # ======= Generator model ==========
        # Do note update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False
        # Perceptors
        self.perceptorA = self.build_perceptorA()
        weights_pathA = '/data/deeplearn/HybridDetectionGAN/ssd512_Clear.h5'
        self.perceptorA.load_weights(weights_pathA, by_name=True)
        self.ssd_lossA = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        # Generators
        if self.ofperceptor:
            self.G_A2B = self.modelGenerator_OF(name='G_A2B_model')
            self.G_B2A = self.modelGenerator_OF(name='G_B2A_model')
            real_A = Input(shape=self.img_shape, name='real_A')
            real_B = Input(shape=self.img_shape, name='real_B')
            synthetic_B = self.G_A2B(real_A)
            synthetic_A = self.G_B2A(real_B)
            dA_guess_synthetic = self.D_A_static(synthetic_A)
            dB_guess_synthetic = self.D_B_static(synthetic_B)
            reconstructed_A = self.G_B2A(synthetic_B)
            valid_percetorA = self.perceptorA(reconstructed_A)
            reconstructed_B = self.G_A2B(synthetic_A)
            model_outputs = [reconstructed_A, reconstructed_B]
            compile_losses = [self.cycle_loss, self.cycle_loss, self.lse, self.lse, self.ssd_lossA.compute_loss]
            compile_weights = [self.lambda_1, self.lambda_2, self.lambda_D, self.lambda_D, self.lambda_P]
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)
            model_outputs.append(valid_percetorA)
            self.G_model = Model(inputs=[real_A, real_B], outputs=model_outputs, name='G_model')
            self.G_model.compile(optimizer=self.opt_G, loss=compile_losses, loss_weights=compile_weights)
        else:
            self.G_A2B = self.modelHybridGenerator(name='G_A2B_model')
            self.G_B2A = self.modelGenerator(name='G_B2A_model')
            real_A = Input(shape=self.img_shape, name='real_A')
            depth_A = Input(shape=(512, 512, 1), name='depth_A')
            real_B = Input(shape=self.img_shape, name='real_B')
            synthetic_B = self.G_A2B([real_A, depth_A])
            synthetic_A, synthetic_A_depth = self.G_B2A(real_B)
            dA_guess_synthetic = self.D_A_static([synthetic_A, synthetic_A_depth])
            dB_guess_synthetic = self.D_B_static(synthetic_B)
            reconstructed_A, reconstructed_A_depth = self.G_B2A(synthetic_B)
            valid_percetorA = self.perceptorA(reconstructed_A)
            reconstructed_B = self.G_A2B([synthetic_A, synthetic_A_depth])
            model_outputs = [reconstructed_A, reconstructed_A_depth, reconstructed_B]
            compile_losses = [self.cycle_loss, self.cycle_loss, self.cycle_loss, self.lse, self.lse, self.ssd_lossA.compute_loss]
            compile_weights = [self.lambda_1, self.lambda_depth, self.lambda_2, self.lambda_D, self.lambda_D, self.lambda_P]
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)
            model_outputs.append(valid_percetorA)
            self.G_model = Model(inputs=[real_A, depth_A, real_B], outputs=model_outputs, name='G_model')
            self.G_model.compile(optimizer=self.opt_G, loss=compile_losses, loss_weights=compile_weights)

        # ======= Data ==========
        sys.stdout.flush()
        if self.use_data_generator:
            print('--- Using dataloader during training ---')
            self.data_generator = load_data.load_data(
                nr_of_channels=self.channels, batch_size=self.batch_size, generator=True, subfolder=image_folder)
            self.data_generator_test = load_data.load_data(
                nr_of_channels=self.channels, batch_size=self.batch_size, generator=False, subfolder=image_folder)
            print('Data has been loaded')
        else:
            print('--- Caching data ---')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.writeMetaDataToJSON()
        # ======= Avoid pre-allocating GPU memory ==========
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        init = K.tf.global_variables_initializer()
        K.get_session().run(init)
        # ======= Initialize training ==========
        sys.stdout.flush()
        # self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        self.load_model_and_generate_synthetic_images()

#===============================================================================
# Architecture functions
    def ck(self, x, k, use_normalization):
        x = Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x
    def c7Ak(self, x, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x
    def dk(self, x, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x
    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = ReflectionPadding2D((1,1))(x0)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        # second layer
        x = ReflectionPadding2D((1, 1))(x)
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = add([x, x0])
        return x
    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x
#===============================================================================
# Models
    def modelDiscriminatorA_OF(self, name=None): #DiscriminatorA for Object-Focus detector
        input_img = Input(shape=self.img_shape)
        x = self.ck(input_img, 64, False)
        x = self.ck(x, 128, True)
        x = self.ck(x, 256, True)
        x = self.ck(x, 512, True)
        # Output layer
        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)
    def modelDiscriminatorA(self, name=None):
        # Specify input
        input_img = Input(shape=self.img_shape)
        depth_img = Input(shape=(512, 512, 1))
        x = Concatenate(axis=-1)([input_img, depth_img])
        x = self.ck(x, 64, False)
        x = self.ck(x, 128, True)
        x = self.ck(x, 256, True)
        x = self.ck(x, 512, True)
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=[input_img, depth_img], outputs=x, name=name)

    def modelDiscriminatorB(self, name=None):
        input_img = Input(shape=self.img_shape)
        x = self.ck(input_img, 64, False)
        x = self.ck(x, 128, True)
        x = self.ck(x, 256, True)
        x = self.ck(x, 512, True)
        x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        x = Activation('sigmoid')(x)
        return Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator_OF(self, name=None):
        input_img = Input(shape=self.img_shape)
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        x = self.dk(x, 64)
        x = self.dk(x, 128)
        for _ in range(4, 13):
            x = self.Rk(x)
        x = self.uk(x, 64)
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        rgb = Activation('tanh')(x)
        return Model(inputs=input_img, outputs=rgb, name=name)

    def modelHybridGenerator(self, name=None):
        # Specify input
        depth_img = Input(shape=(512, 512, 1))
        input_img = Input(shape=self.img_shape)
        alpha_R = alphaRLayer(output_dim=(512, 512, 1))(depth_img)
        alpha_G = alphaGLayer(output_dim=(512, 512, 1))(depth_img)
        alpha_B = alphaBLayer(output_dim=(512, 512, 1))(depth_img)
        transmission = Concatenate(axis=-1)([alpha_R, alpha_G, alpha_B])
        direct = Multiply()([input_img, transmission])
        # backscatter
        bsbeta_R = bsbetaRLayer(output_dim=(512, 512, 1))(depth_img)
        bsbeta_G = bsbetaRLayer(output_dim=(512, 512, 1))(depth_img)
        bsbeta_B = bsbetaRLayer(output_dim=(512, 512, 1))(depth_img)
        negtransmission = Concatenate(axis=-1)([bsbeta_R, bsbeta_G, bsbeta_B])
        backscatter = bglightLayer(output_dim=(512, 512, 3))(negtransmission)
        physical = Add()([direct, backscatter])

        x = self.c7Ak(input_img, 32)
        x = self.dk(x, 64)
        x = self.dk(x, 128)
        for _ in range(4, 13):
            x = self.Rk(x)
        x = self.uk(x, 64)
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        x = Conv2D(512, kernel_size=3, strides=1)(x)
        x = Concatenate(axis=-1)([x, physical])
        x = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
        return Model(inputs=[input_img, depth_img], outputs=x, name=name)

    def modelGenerator(self, name=None):
        input_img = Input(shape=self.img_shape)
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 32)
        x = self.dk(x, 64)
        x = self.dk(x, 128)
        for _ in range(4, 13):
            x = self.Rk(x)
        x = self.uk(x, 64)
        x = self.uk(x, 32)
        x = ReflectionPadding2D((3, 3))(x)
        rgb = Conv2D(self.channels, kernel_size=7, strides=1)(x)
        # rgb = Activation('tanh')(rgb)
        depth = Conv2D(1, kernel_size=7, strides=1)(x)
        # depth = Activation('tanh')(depth)
        return Model(inputs=input_img, outputs=[rgb, depth], name=name)

    def build_perceptorA(self):

        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
        img_A = Input(shape=self.img_shape)
        ssd_512_outputA = ssd_512(img_A,
                                 n_classes=self.n_classes,
                                 mode='training',
                                 l2_regularization=0.0005,
                                 scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=True,
                                 steps=[8, 16, 32, 64, 128, 256, 512],
                                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 clip_boxes=False,
                                 variances=[0.1, 0.1, 0.2, 0.2],
                                 normalize_coords=True,
                                 subtract_mean=[123, 117, 104],
                                 swap_channels=[2, 1, 0])

        return Model(img_A, ssd_512_outputA)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1, save_interval=1):
        def parse_xmlA(image_id):
            ssd_input_encoder = SSDInputEncoder(img_height=512,
                                                img_width=512,
                                                n_classes=self.n_classes,
                                                predictor_sizes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
                                                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                                                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                                         [1.0, 2.0, 0.5],
                                                                         [1.0, 2.0, 0.5]],
                                                two_boxes_for_ar1=True,
                                                steps=[8, 16, 32, 64, 128, 256, 512],
                                                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                                clip_boxes=False,
                                                variances=[0.1, 0.1, 0.2, 0.2],
                                                matching_type='multi',
                                                pos_iou_threshold=0.5,
                                                neg_iou_limit=0.5,
                                                normalize_coords=True)
            classes = ['background', 'bowl', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']
            AnnotationPath='/data/deeplearn/VOCdevkit/MultiView/Annotations'
            labels = []

            with open(os.path.join(AnnotationPath, image_id + '.xml')) as f:
                soup = BeautifulSoup(f, 'xml')
            boxes = []
            objects = soup.find_all('object')
            for obj in objects:
                box = []
                class_name = obj.find('name', recursive=False).text
                class_id = classes.index(class_name)
                bndbox = obj.find('bndbox', recursive=False)
                xmin = int(bndbox.xmin.text)
                ymin = int(bndbox.ymin.text)
                xmax = int(bndbox.xmax.text)
                ymax = int(bndbox.ymax.text)
                box.append(class_id)
                box.append(xmin)
                box.append(ymin)
                box.append(xmax)
                box.append(ymax)
                boxes.append(box)
            labels.append(boxes)
            batch_y = deepcopy(labels)
            batch_y=np.asarray(batch_y)
            batch_y_encoded = ssd_input_encoder(batch_y, diagnostics=False)
            return batch_y_encoded

        def run_training_iteration(loop_index, epoch_iterations):
            if self.ofperceptor:
                # ======= Discriminator training ==========
                synthetic_images_B = self.G_A2B.predict(real_images_A)
                synthetic_images_A = self.G_B2A.predict(real_images_B)
                synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
                synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)
                for _ in range(self.discriminator_iterations):
                    DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                    DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                    DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                    DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                    D_loss = DA_loss + DB_loss
                    if self.discriminator_iterations > 1:
                        sys.stdout.flush()
                # ======= Generator training ==========
                target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
                target_data.append(ones)
                target_data.append(ones)
                target_data.append(ssd_512_outputA)
                for _ in range(self.generator_iterations):
                    G_loss = self.G_model.train_on_batch(
                        x=[real_images_A, real_images_B], y=target_data)
                    if self.generator_iterations > 1:
                        sys.stdout.flush()
            else:
                synthetic_images_B = self.G_A2B.predict([real_images_A, depth_images_A])
                synthetic_images_A, synthetic_images_A_depth = self.G_B2A.predict(real_images_B)
                synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
                synthetic_images_A_depth = synthetic_pool_A_depth.query(synthetic_images_A_depth)
                synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)
                for _ in range(self.discriminator_iterations):
                    DA_loss_real = self.D_A.train_on_batch(x=[real_images_A, depth_images_A], y=ones)
                    DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                    DA_loss_synthetic = self.D_A.train_on_batch(x=[synthetic_images_A, synthetic_images_A_depth], y=zeros)
                    DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                    D_loss = DA_loss + DB_loss
                    if self.discriminator_iterations > 1:
                        sys.stdout.flush()
                # ======= Generator training ==========
                target_data = [real_images_A, depth_images_A, real_images_B] #Compare reconstructed images to real images
                target_data.append(ones)
                target_data.append(ones)
                target_data.append(ssd_512_outputA)
                for _ in range(self.generator_iterations):
                    G_loss = self.G_model.train_on_batch(x=[real_images_A, depth_images_A, real_images_B], y=target_data)
                    if self.generator_iterations > 1:
                        sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]
            perception_loss_A = G_loss[5]

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)
            perception_losses_A.append(perception_loss_A)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A + perception_losses_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('Perceptual_loss: ', perception_loss_A)
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)
        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()
        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []
        perception_losses_A = []
        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []
        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_A_depth = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)
        # labels
        label_shape = (batch_size,) + self.D_A.output_shape[1:]
        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0
        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()
        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_imgs_A = images[0]
                    real_imgs_B = images[1]
                    real_imname_A = images[2]
                    real_imname_B = images[3]

                    real_images_A = real_imgs_A[:, :, :, 0:3]
                    depth_images_A = real_imgs_A[:, :, :, 3]
                    depth_images_A = depth_images_A[:, :, :, np.newaxis]

                    real_images_B = real_imgs_B[:, :, :, 0:3]
                    ssd_512_outputA = parse_xmlA(images[2])

                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]
                    # Run all training steps
                    run_training_iteration(loop_index, self.data_generator.__len__())
                    # Store models
                    if loop_index % 2000 == 0:
                        self.saveModel(self.D_A, epoch, loop_index)
                        self.saveModel(self.D_B, epoch, loop_index)
                        self.saveModel(self.G_A2B, epoch, loop_index)
                        self.saveModel(self.G_B2A, epoch, loop_index)
                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index += 1
            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)
            # Flush out prints each loop iteration
            sys.stdout.flush()
#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        toimage(image, cmin=-1, cmax=1).save(path_name)

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.A_train), len(self.B_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        K.set_value(model.optimizer.lr, new_lr)
#===============================================================================
# Save and load
    def saveModel(self, model, epoch, loop_index):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path_w = 'saved_models/{}/{}_weights_epoch_{}_loop_{}.hdf5'.format(self.date_time, model.name, epoch, loop_index)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}_loop_{}.json'.format(self.date_time, model.name, epoch, loop_index)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'lambda_depth': self.lambda_depth,
            'lambda_perceptor': self.lambda_P,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'Structure':'ReflectionPadding2D((3, 3)), Conv2D(512, kernel_size=3, strides=1), Concatenate(axis=-1)([x, physical]), Conv2D(3, kernel_size=3, strides=1, padding=same), ModelGenerator no tanh',
            'REAL_LABEL': self.REAL_LABEL,
        })
        with open('images/{}/perception_meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        path_to_weights = os.path.join('saved_models', '{}_weights.hdf5'.format(model.name))
        model.load_weights(path_to_weights)

    def load_model_and_generate_synthetic_images(self):
        self.load_model_and_weights(self.G_A2B)
        self.load_model_and_weights(self.G_B2A)
        def save_image(image, name):
            if 'depth' in name:
                image = image[:, :, 0]
            toimage(image, cmin=-1, cmax=1).save(os.path.join(
                'Results', name))
        if self.use_data_generator:
            for images in self.data_generator_test:
                real_images_A=images[0]
                real_images_B=images[1]
                real_imname_A=images[2]
                real_imname_B=images[3]
                if self.ofperceptor:
                    synthetic_images_A = self.G_B2A.predict(real_images_B[:, :, :, 0:3])
                    name = real_imname_B + '.png'
                    synt_A = synthetic_images_A[0]
                    save_image(synt_A, name)
                else:
                    synthetic_images_A = self.G_B2A.predict(real_images_B[:, :, :, 0:3])
                    name = real_imname_B + '.png'
                    synt_A = synthetic_images_A[0][0]
                    save_image(synt_A, name)


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):

        newshape = (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])
        return newshape

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

if __name__ == '__main__':
    GAN = CycleGAN()
