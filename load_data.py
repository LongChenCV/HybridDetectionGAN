import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread


def load_data(nr_of_channels, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0):

    trainA_path = '/data/deeplearn/VOCdevkit/MultiView/trainJPEGImages512'
    trainB_path = '/data/deeplearn/VOCdevkit/chinamm2019uw/chinamm2019uw_train/JPEGImages512'
    testA_path = '/data/deeplearn/HybridDetectionGAN/dataset/MultiView/JPEGImages512'
    testB_path = '/data/deeplearn/HybridDetectionGAN/dataset/ChinaMM/JPEGImages512'
    trainA_image_names = os.listdir(trainA_path)
    if nr_A_train_imgs != None:
        trainA_image_names = trainA_image_names[:nr_A_train_imgs]

    trainB_image_names = os.listdir(trainB_path)
    if nr_B_train_imgs != None:
        trainB_image_names = trainB_image_names[:nr_B_train_imgs]

    testA_image_names = os.listdir(testA_path)
    if nr_A_test_imgs != None:
        testA_image_names = testA_image_names[:nr_A_test_imgs]

    testB_image_names = os.listdir(testB_path)
    if nr_B_test_imgs != None:
        testB_image_names = testB_image_names[:nr_B_test_imgs]

    if generator:
        return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        return data_sequence_test(testA_path, testB_path, testA_image_names, testB_image_names, batch_size=batch_size)


def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)).resize((512, 512)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> street view
                image = np.array(Image.open(os.path.join(image_path, image_name)).resize((512, 512)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)


def create_depthimage_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)).resize((512, 512)))
                realimage = image[:, :, np.newaxis]
            else:                   # RGB image -> street view
                realimage = np.array(Image.open(os.path.join(image_path, image_name)).resize((512, 512)))
                depth_image_path = os.path.join(image_path, image_name)
                depth_image_path = depth_image_path.replace("JPEGImages512", "DepthData512")
                depth_image = np.array(Image.open(depth_image_path).resize((512, 512)))
                # depth_image = np.array(Image.open(os.path.join(depth_image_path, image_name)))
                depth_image = depth_image[:, :, np.newaxis]
            realimage = normalize_array(realimage)
            depth_image = normalize_array(depth_image)
            image = np.concatenate((realimage, depth_image), axis=-1)
            image_array.append(image)

    return np.array(image_array)

# If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
    return array


class data_sequence(Sequence):

    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_imname_A = batch_A[0].split('/')[-1].split('.')[-2]
        real_imname_B = batch_B[0].split('/')[-1].split('.')[-2]
        real_images_A = create_depthimage_array(batch_A, '', 3)
        real_images_B = create_image_array(batch_B, '', 3)
        return real_images_A, real_images_B, real_imname_A, real_imname_B  # input_data, target_data

class data_sequence_test(Sequence):

    def __init__(self, testA_path, testB_path, image_list_A, image_list_B, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.test_A = []
        self.test_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.test_A.append(os.path.join(testA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.test_B.append(os.path.join(testB_path, image_name))

    def __len__(self):
        return int(max(len(self.test_A), len(self.test_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.test_A), len(self.test_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.test_A) <= len(self.test_B):
                indexes_A = np.random.randint(len(self.test_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.test_A[i])
                batch_B = self.test_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.test_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.test_B[i])
                batch_A = self.test_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.test_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.test_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_imname_A = batch_A[0].split('/')[-1].split('.')[-2]
        real_imname_B = batch_B[0].split('/')[-1].split('.')[-2]
        real_images_A = create_depthimage_array(batch_A, '', 3)
        real_images_B = create_image_array(batch_B, '', 3)
        return real_images_A, real_images_B, real_imname_A, real_imname_B

if __name__ == '__main__':
    load_data()
