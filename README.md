# HybridDetectionGAN
This is the source code of the paper "Perceptual underwater image enhancement with deep learning and physical priors"

Long Chen, Zheheng Jiang, Lei Tong, Zhihua Liu, Aite Zhao, Qianni Zhang, Junyu Dong, and Huiyu. "Perceptual underwater image enhancement with deep learning and physical priors".

---
### Abstract
Underwater image enhancement, as a pre-processing step to support the following object detection task, has drawn considerable attention in the field of underwater navigation and ocean exploration. However, most of the existing underwater image enhancement strategies tend to consider enhancement and detection as two fully independent modules with no interaction, and the practice of separate optimisation does not always help the following object detection task. In this paper, we propose two perceptual enhancement models, each of which uses a deep enhancement model with a detection perceptor. The detection perceptor provides feedback information in the form of gradients to guide the enhancement model to generate patch level visually pleasing or detection favourable images. In addition, due to the lack of training data, a hybrid underwater image synthesis model, which fuses physical priors and data-driven cues, is proposed to synthesise training data and generalise our enhancement model for real-world underwater images. Experimental results show the superiority of our proposed method over several state-of-the-art methods on both real-world and synthetic underwater datasets

### Dependencies
* Keras and python 3

### Datasets
* ChinaMM dataset: https://rwenqi.github.io/chinaMM2019uw/
* MultiView dataset: http://rgbd-dataset.cs.washington.edu/dataset/rgbd-scenes/

### Test 
* Download our well-trained HybridDetectionGAN models from https://1drv.ms/u/s!At3lDmLw-VwAu0qU-PWQ52pUiLHQ?e=ELzeGk and put them in ../saved_models.
* Set the test mode in model_perception.py:
Comment the test commond self.load_model_and_generate_synthetic_images(),
Uncomment the train command: self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
* Run python model_perception.py
### Train 
* First train a high capacity detector using the source code in ../ssd_keras on the MultiView dataset or other detection datasets containing high quality images and depth images, then save the well-trained detection model as ssd512_Clear.h5 in ../HybridDetectionGAN/
* If you use your own dataset, please set the variables: classes and AnnotationPath as your own dataset in model_perception.py.
* Set the train mode in model_perception.py:
Comment the test commond self.load_model_and_generate_synthetic_images(),
Uncomment the train command: self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
* Run python model_perception.py
