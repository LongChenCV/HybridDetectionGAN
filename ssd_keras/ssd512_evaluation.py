from keras import backend as K
from keras.optimizers import Adam
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator
# Set a few configuration parameters.
img_height = 512
img_width = 512
n_classes = 5
model_mode = 'inference'
K.clear_session()   # Clear previous models from memory.
model = ssd_512(image_size=(img_height, img_width, 3),
                n_classes=n_classes,
                mode=model_mode,
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
             [1.0, 2.0, 0.5],
             [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 128, 256, 512],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.
weights_path = '/scratch/deeplearn/lc408/ssd_keras/ssd512_Clear_epoch-01.h5'
model.load_weights(weights_path, by_name=True)
# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = DataGenerator()
# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = '/data/deeplearn/VOCdevkit/MultiView/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = '/data/deeplearn/VOCdevkit/MultiView/Annotations/'
Pascal_VOC_dataset_image_set_filename = '/data/deeplearn/VOCdevkit/MultiView/ImageSets/Main/test.txt'

classes = ['background', 'bowl', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']
dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)
results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)
mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
