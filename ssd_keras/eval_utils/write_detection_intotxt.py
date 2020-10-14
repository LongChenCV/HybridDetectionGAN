'''
An evaluator to compute the Pascal VOC-style mean average precision (both the pre-2010
and post-2010 algorithm versions) of a given Keras SSD model on a given dataset.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from math import ceil
from tqdm import trange
import sys
import warnings
import os
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
import cv2
from bounding_box_utils.bounding_box_utils import iou

class Evaluator:
    '''
    Computes the mean average precision of the given Keras SSD model on the given dataset.

    Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
    and post-2010 (integration) algorithm versions.

    Optionally also returns the average precisions, precisions, and recalls.

    The algorithm is identical to the official Pascal VOC pre-2010 detection evaluation algorithm
    in its default settings, but can be cusomized in a number of ways.
    '''

    def __init__(self,
                 imdir,
                 model,
                 n_classes,
                 data_generator,
                 method,
                 model_mode='inference',
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5},
                 gt_format={'class_id': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5}):
        '''
        Arguments:
            model (Keras model): A Keras SSD model object.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            data_generator (DataGenerator): A `DataGenerator` object with the evaluation dataset.
            model_mode (str, optional): The mode in which the model was created, i.e. 'training', 'inference' or 'inference_fast'.
                This is needed in order to know whether the model output is already decoded or still needs to be decoded. Refer to
                the model documentation for the meaning of the individual modes.
            pred_format (dict, optional): A dictionary that defines which index in the last axis of the model's decoded predictions
                contains which bounding box coordinate. The dictionary must map the keywords 'class_id', 'conf' (for the confidence),
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis.
            gt_format (list, optional): A dictionary that defines which index of a ground truth bounding box contains which of the five
                items class ID, xmin, ymin, xmax, ymax. The expected strings are 'xmin', 'ymin', 'xmax', 'ymax', 'class_id'.
        '''

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")
        self.imdir=imdir
        self.model = model
        self.data_generator = data_generator
        self.n_classes = n_classes
        self.model_mode = model_mode
        self.method = method
        self.pred_format = pred_format
        self.gt_format = gt_format
        self.prediction_results = None

    def __call__(self,
                 img_height,
                 img_width,
                 batch_size,
                 data_generator_mode='resize',
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='sample',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 return_precisions=False,
                 return_recalls=False,
                 return_average_precisions=False,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):
        #############################################################################################
        # Write detections into txt per image
        #############################################################################################

        # self.predict_on_dataset(img_height=img_height,
        #                         img_width=img_width,
        #                         batch_size=batch_size,
        #                         data_generator_mode=data_generator_mode,
        #                         decoding_confidence_thresh=decoding_confidence_thresh,
        #                         decoding_iou_threshold=decoding_iou_threshold,
        #                         decoding_top_k=decoding_top_k,
        #                         decoding_pred_coords=decoding_pred_coords,
        #                         decoding_normalize_coords=decoding_normalize_coords,
        #                         decoding_border_pixels=border_pixels,
        #                         round_confidences=round_confidences,
        #                         verbose=verbose,
        #                         ret=False)
        #############################################################################################
        # Write predictions into txt per image
        #############################################################################################
        self.write_detection_perim(img_height=img_height,
                                img_width=img_width,
                                batch_size=batch_size,
                                data_generator_mode=data_generator_mode,
                                decoding_confidence_thresh=decoding_confidence_thresh,
                                decoding_iou_threshold=decoding_iou_threshold,
                                decoding_top_k=decoding_top_k,
                                decoding_pred_coords=decoding_pred_coords,
                                decoding_normalize_coords=decoding_normalize_coords,
                                decoding_border_pixels=border_pixels,
                                round_confidences=round_confidences,
                                verbose=verbose,
                                ret=False)

        #############################################################################################
        # Write predictions into txt per class
        #############################################################################################
        classes = ['background', 'seacucumber', 'seaurchin', 'scallop', 'starfish']
        # classes = ['background', 'stone']
        # classes = ['background', 'bowl', 'cap', 'cereal_box', 'coffee_mug', 'soda_can']
        # self.write_predictions_perclass(classes, out_file_prefix=str(self.methodindex)+'comp4_det_test_', verbose=True)
    def predict_on_dataset(self,
                           img_height,
                           img_width,
                           batch_size,
                           data_generator_mode='resize',
                           decoding_confidence_thresh=0.01,
                           decoding_iou_threshold=0.45,
                           decoding_top_k=200,
                           decoding_pred_coords='centroids',
                           decoding_normalize_coords=True,
                           decoding_border_pixels='include',
                           round_confidences=False,
                           verbose=True,
                           ret=False):
        class_id_pred = self.pred_format['class_id']
        conf_pred = self.pred_format['conf']
        xmin_pred = self.pred_format['xmin']
        ymin_pred = self.pred_format['ymin']
        xmax_pred = self.pred_format['xmax']
        ymax_pred = self.pred_format['ymax']

        #############################################################################################
        # Configure the data generator for the evaluation.
        #############################################################################################

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height,width=img_width, labels_format=self.gt_format)
        if data_generator_mode == 'resize':
            transformations = [convert_to_3_channels,
                               resize]
        elif data_generator_mode == 'pad':
            random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width/img_height, labels_format=self.gt_format)
            transformations = [convert_to_3_channels,
                               random_pad,
                               resize]
        else:
            raise ValueError("`data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(data_generator_mode))

        # Set the generator parameters.
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 transformations=transformations,
                                                 label_encoder=None,
                                                 returns={'processed_images',
                                                          'image_ids',
                                                          'evaluation-neutral',
                                                          'inverse_transform',
                                                          'original_labels'},
                                                 keep_images_without_gt=True,
                                                 degenerate_box_handling='remove')

        # If we don't have any real image IDs, generate pseudo-image IDs.
        # This is just to make the evaluator compatible both with datasets that do and don't
        # have image IDs.
        if self.data_generator.image_ids is None:
            self.data_generator.image_ids = list(range(self.data_generator.get_dataset_size()))

        #############################################################################################
        # Predict over all batches of the dataset and store the predictions.
        #############################################################################################

        # We have to generate a separate results list for each class.
        results = [list() for _ in range(self.n_classes + 1)]

        # Create a dictionary that maps image IDs to ground truth annotations.
        # We'll need it below.
        image_ids_to_labels = {}

        # Compute the number of batches to iterate over the entire dataset.
        n_images = self.data_generator.get_dataset_size()
        n_batches = int(ceil(n_images / batch_size))
        if verbose:
            print("Number of images in the evaluation dataset: {}".format(n_images))
            print()
            tr = trange(n_batches, file=sys.stdout)
            tr.set_description('Producing predictions batch-wise')
        else:
            tr = range(n_batches)

        # Loop over all batches.
        for j in tr:
            # Generate batch.
            batch_X, batch_image_ids, batch_eval_neutral, batch_inverse_transforms, batch_orig_labels = next(generator)
            # Predict.
            y_pred = self.model.predict(batch_X)
            # If the model was created in 'training' mode, the raw predictions need to
            # be decoded and filtered, otherwise that's already taken care of.
            if self.model_mode == 'training':
                # Decode.
                y_pred = decode_detections(y_pred,
                                           confidence_thresh=decoding_confidence_thresh,
                                           iou_threshold=decoding_iou_threshold,
                                           top_k=decoding_top_k,
                                           input_coords=decoding_pred_coords,
                                           normalize_coords=decoding_normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           border_pixels=decoding_border_pixels)
            else:
                # Filter out the all-zeros dummy elements of `y_pred`.
                y_pred_filtered = []
                for i in range(len(y_pred)):
                    y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
                y_pred = y_pred_filtered
            # Convert the predicted box coordinates for the original images.
            y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

            for k, batch_item in enumerate(y_pred):
                image_id = batch_image_ids[k]
                for box in batch_item:
                    class_id = int(box[class_id_pred])
                    # Round the box coordinates to reduce the required memory.
                    if round_confidences:
                        confidence = round(box[conf_pred], round_confidences)
                    else:
                        confidence = box[conf_pred]
                    xmin = round(box[xmin_pred], 1)
                    ymin = round(box[ymin_pred], 1)
                    xmax = round(box[xmax_pred], 1)
                    ymax = round(box[ymax_pred], 1)
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    results[class_id].append(prediction)
                self.prediction_results = results
        print("All results per image saved .")

    def write_detection_perim(self,
                           img_height,
                           img_width,
                           batch_size,
                           data_generator_mode='resize',
                           decoding_confidence_thresh=0.01,
                           decoding_iou_threshold=0.45,
                           decoding_top_k=200,
                           decoding_pred_coords='centroids',
                           decoding_normalize_coords=True,
                           decoding_border_pixels='include',
                           round_confidences=False,
                           verbose=True,
                           ret=False):
        class_id_pred = self.pred_format['class_id']
        conf_pred = self.pred_format['conf']
        xmin_pred = self.pred_format['xmin']
        ymin_pred = self.pred_format['ymin']
        xmax_pred = self.pred_format['xmax']
        ymax_pred = self.pred_format['ymax']

        #############################################################################################
        # Configure the data generator for the evaluation.
        #############################################################################################

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width, labels_format=self.gt_format)
        if data_generator_mode == 'resize':
            transformations = [convert_to_3_channels,
                               resize]
        elif data_generator_mode == 'pad':
            random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width / img_height, labels_format=self.gt_format)
            transformations = [convert_to_3_channels,
                               random_pad,
                               resize]
        else:
            raise ValueError("`data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(
                data_generator_mode))

        # Set the generator parameters.
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 transformations=transformations,
                                                 label_encoder=None,
                                                 returns={'processed_images',
                                                          'image_ids',
                                                          'evaluation-neutral',
                                                          'inverse_transform',
                                                          'original_labels'},
                                                 keep_images_without_gt=True,
                                                 degenerate_box_handling='remove')

        # If we don't have any real image IDs, generate pseudo-image IDs.
        # This is just to make the evaluator compatible both with datasets that do and don't
        # have image IDs.
        if self.data_generator.image_ids is None:
            self.data_generator.image_ids = list(range(self.data_generator.get_dataset_size()))

        #############################################################################################
        # Predict over all batches of the dataset and store the predictions.
        #############################################################################################

        # We have to generate a separate results list for each class.
        results = [list() for _ in range(self.n_classes + 1)]

        # Create a dictionary that maps image IDs to ground truth annotations.
        # We'll need it below.
        image_ids_to_labels = {}

        # Compute the number of batches to iterate over the entire dataset.
        n_images = self.data_generator.get_dataset_size()
        n_batches = int(ceil(n_images / batch_size))
        if verbose:
            print("Number of images in the evaluation dataset: {}".format(n_images))
            print()
            tr = trange(n_batches, file=sys.stdout)
            tr.set_description('Producing predictions batch-wise')
        else:
            tr = range(n_batches)

        # Loop over all batches.
        for j in tr:
            # Generate batch.
            batch_X, batch_image_ids, batch_eval_neutral, batch_inverse_transforms, batch_orig_labels = next(
                generator)
            # Predict.
            y_pred = self.model.predict(batch_X)
            # If the model was created in 'training' mode, the raw predictions need to
            # be decoded and filtered, otherwise that's already taken care of.
            if self.model_mode == 'training':
                # Decode.
                y_pred = decode_detections(y_pred,
                                           confidence_thresh=decoding_confidence_thresh,
                                           iou_threshold=decoding_iou_threshold,
                                           top_k=decoding_top_k,
                                           input_coords=decoding_pred_coords,
                                           normalize_coords=decoding_normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           border_pixels=decoding_border_pixels)
            else:
                # Filter out the all-zeros dummy elements of `y_pred`.
                y_pred_filtered = []
                for i in range(len(y_pred)):
                    y_pred_filtered.append(y_pred[i][y_pred[i, :, 0] != 0])
                y_pred = y_pred_filtered
            # Convert the predicted box coordinates for the original images.
            y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

            for k, batch_item in enumerate(y_pred):
                image_id = batch_image_ids[k]
                txtpath = self.imdir + image_id + '.txt'
                if not os.path.exists(txtpath):
                    os.mknod(txtpath)
                img = cv2.imread(self.imdir+image_id+'.png')
                file_fid = open(txtpath, 'w')
                for box in batch_item:
                    class_id = int(box[class_id_pred])
                    if round_confidences:
                        confidence = round(box[conf_pred], round_confidences)
                    else:
                        confidence = box[conf_pred]
                    if confidence < 0.5:
                        continue
                    xmin = int(round(box[xmin_pred], 1))
                    ymin = int(round(box[ymin_pred], 1))
                    xmax = int(round(box[xmax_pred], 1))
                    ymax = int(round(box[ymax_pred], 1))
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    if class_id == 1:
                        class_name = 'seacucumber'
                    if class_id == 2:
                        class_name = 'seaurchin'
                    if class_id == 3:
                        class_name = 'scallop'
                    # if class_id == 4:
                    #     class_name = 'starfish'
                    boxstr = class_name + ' ' + str(confidence) + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(
                        xmax) + ' ' + str(ymax)
                    file_fid.write(boxstr + '\n')

                    if class_name == 'seacucumber':
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, class_name, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    if class_name == 'seaurchin':
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.putText(img, class_name, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    if class_name == 'scallop':
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                        cv2.putText(img, class_name, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                    results[class_id].append(prediction)
                self.prediction_results = results
                file_fid.close()
                cv2.imwrite(self.imdir+'GT' + image_id+'.png', img)
        print("All results per image saved .")

    def write_predictions_perclass(self,
                                 classes=None,
                                 out_file_prefix='comp3_det_test_',
                                 verbose=True):

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        # We generate a separate results file for each class.
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Writing results file for class {}/{}.".format(class_id, self.n_classes))

            if classes is None:
                class_suffix = '{:04d}'.format(class_id)
            else:
                class_suffix = classes[class_id]

            results_file = open('{}{}.txt'.format(out_file_prefix, class_suffix), 'w')

            for prediction in self.prediction_results[class_id]:

                prediction_list = list(prediction)
                prediction_list[0] = '{}'.format(prediction_list[0])
                prediction_list[1] = round(prediction_list[1], 4)
                prediction_txt = ' '.join(map(str, prediction_list)) + '\n'
                results_file.write(prediction_txt)

            results_file.close()
            print("All results per class saved.")







