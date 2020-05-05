# Written by: Sharif Elguindi, MS, DABR
# ==============================================================================
#
# This script returns PNG image files and associated masks for 2D training of images
# using FCN architectures in tensorflow.
#
# Usage:
#
#   python dicomrt_to_traindata.py \
#   --numShards='number of pieces to break dataset into for transfer purposes
#   --rawDir='path\to\data\'
#   --saveDir='path\to\save\'
#   --datasetName='structure_name_to_search_for'

from __future__ import print_function

import pickle

import numpy as np
import tensorflow as tf
import h5py
import sys
import glob
import math
import build_data
import scipy.io as sio
from imgAug import *
from functions import find

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('jue', False,
                     'Set to true to collect data based on .mat CERR files.')

flags.DEFINE_boolean('MHD', False,
                     'Set to true to collect data based on .mat CERR files.')

flags.DEFINE_integer('numShards', 100,
                     'Split train/val data into chucks if large dateset >2-3000 (default, 1)')

flags.DEFINE_string('rawDir', 'G:\\Projects\\DicomToMask\\datasets\\MR_RAW\\',
                    'absolute path to where raw data is collected from.')

flags.DEFINE_string('saveDir', 'G:\\Projects\\DicomToMask\\datasets\\',
                    'absolute path to where processed data is saved.')

flags.DEFINE_string('datasetName', 'ProstateUpdate_16_99_PosOnly_Aug',
                    'string name of structure to export')

def create_tfrecord(structure_path):

    planeList = ['ax', 'cor', 'sag']
    planeDir = ['Axial', 'Coronal', 'Sag']
    filename_train = 'train_'
    filename_val = 'val_'
    i = 0
    for plane in planeList:

        file_base = os.path.join(structure_path, 'processed', 'ImageSets', planeDir[i])
        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_train + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        path = os.path.join(structure_path, 'processed', 'PNGImages')
        pattern = plane + '*.png'
        files = find(pattern, path)
        for file in files:
            if file.find(plane) > 0 \
                    and (file.find(plane + '1011_') < 1 and
                         file.find(plane + '1511_') < 1 and
                         file.find(plane + '2011_') < 1 and
                         file.find(plane + '2511_') < 1 and
                         file.find(plane + '3011_') < 1 and
                         file.find(plane + '3511_') < 1 and
                         file.find(plane + '4011_') < 1 and
                         file.find(plane + '4511_') < 1 and
                         file.find(plane + '511_') < 1):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_train + plane, k)

        if not os.path.exists(file_base):
            os.makedirs(file_base)
        f = open(os.path.join(file_base, filename_val + plane + '.txt'), 'a')
        f.truncate()
        k = 0
        for file in files:
            if file.find(plane) > 0 \
                    and (file.find(plane + '511_') > 0 or
                         file.find(plane + '1011_') > 0 or
                         file.find(plane + '1511_') > 0 or
                         file.find(plane + '2011_') > 0 or
                         file.find(plane + '2511_') > 0 or
                         file.find(plane + '3011_') > 0 or
                         file.find(plane + '3511_') > 0 or
                         file.find(plane + '4011_') > 0 or
                         file.find(plane + '4511_') > 0):
                h = file.split(os.sep)
                f.write(h[-1].replace('.png','') +'\n')
                k = k + 1
        f.close()
        print(filename_val + plane, k)
        i = i + 1

        dataset_splits = glob.glob(os.path.join(file_base, '*.txt'))
        for dataset_split in dataset_splits:
            _convert_dataset(dataset_split, FLAGS.numShards, structure_path, plane)

    return

def _convert_dataset(dataset_split, _NUM_SHARDS, structure_path, plane):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  image_folder = os.path.join(structure_path, 'processed', 'PNGImages')
  semantic_segmentation_folder = os.path.join(structure_path, 'processed', 'SegmentationClass')
  image_format = label_format = 'png'

  if not os.path.exists(os.path.join(structure_path, 'tfrecord'+ '_' + plane)):
      os.makedirs(os.path.join(structure_path, 'tfrecord'+ '_' + plane))

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)

  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  filenames.sort()
  # random.shuffle(filenames)
  print(filenames)
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        structure_path, 'tfrecord'+ '_' + plane,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            image_folder, filenames[i] + '.' + image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            semantic_segmentation_folder,
            filenames[i] + '.' + label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, str.encode(filenames[i],'utf-8'), height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

# create_tfrecord(os.path.join(FLAGS.saveDir, FLAGS.datasetName))

def main(unused_argv):

    data_path = FLAGS.rawDir
    p_num = 1
    incomplete = []

    if FLAGS.jue:
        patient_sets = find('*.mat', data_path)
        patient_sets.sort()
        for patient in patient_sets:
                data = sio.loadmat(patient)
                data_lens = sio.loadmat(patient.replace('\\data\\','\\Lens_label\\').replace('.mat','_Lens_label.mat'))
                vect_lens = data_lens['vect']

                vect = data['vect']
                scan = vect['img'][0, 0]
                mask = vect['label'][0, 0]
                mask_lens = vect_lens['lens_label'][0,0]

                ## Needed if passing through CERR
                scan_rot = scan.transpose(2,1,0)
                mask_rot = mask.transpose(2,1,0)
                mask_lens_rot = mask_lens.transpose(2,1,0)
                unique, counts = np.unique(mask_lens_rot, return_counts=True)
                vals = dict(zip(unique, counts))
                print(patient)
                print(vals)
                unique, counts = np.unique(mask_rot, return_counts=True)
                vals = dict(zip(unique, counts))
                print(vals)
                # parotids
                np.place(mask_rot, mask_rot == 2, 2)

                # submands
                np.place(mask_rot, mask_rot == 3, 3)
                np.place(mask_rot, mask_rot == 4, 4)

                # bps
                np.place(mask_rot, mask_rot == 5, 5)
                np.place(mask_rot, mask_rot == 6, 6)

                # mandible
                np.place(mask_rot, mask_rot == 7, 7)

                # cord
                np.place(mask_rot, mask_rot == 8, 8)

                # brainstem
                np.place(mask_rot, mask_rot == 9, 9)

                # OC
                np.place(mask_rot, mask_rot == 10, 10)

                # larynx
                np.place(mask_rot, mask_rot == 11, 11)


                # chiasm
                np.place(mask_rot, mask_rot == 12, 12)

                # optics
                np.place(mask_rot, mask_rot == 13, 13)
                np.place(mask_rot, mask_rot == 14, 14)


                # eyes
                np.place(mask_rot, mask_rot == 15, 15)
                np.place(mask_rot, mask_rot == 16, 16)

                # lenses
                np.place(mask_lens_rot, mask_lens_rot == 17, 100)
                np.place(mask_lens_rot, mask_lens_rot == 18, 100)

                np.place(mask_rot, mask_rot == 19, 19)

                mask_rot = mask_lens_rot + mask_rot
                #
                # np.place(mask_rot, mask_rot > 13, 12)
                np.place(mask_rot, mask_rot == 115, 17)
                np.place(mask_rot, mask_rot == 116, 18)
                np.place(mask_rot, mask_rot == 117, 17)
                np.place(mask_rot, mask_rot == 118, 18)

                unique, counts = np.unique(mask_rot, return_counts=True)
                print(unique)
                if len(unique) >= 19:
                    data_export_MR_3D(scan_rot, mask_rot, FLAGS.saveDir, p_num, FLAGS.datasetName)
                    print(p_num)
                    p_num = p_num + 1
                else:
                    incomplete.append(p_num)
        create_tfrecord(os.path.join(FLAGS.saveDir, FLAGS.datasetName))
        with open('incomplete.pickle','wb') as f:
            pickle.dump(incomplete,f)

    elif FLAGS.MHD:
        patient_sets = find('*segmentation.mhd', data_path)
        patient_sets.sort()
        for patient in patient_sets:
            s = sitk.ReadImage(patient.replace('_segmentation', ''))
            m = sitk.ReadImage(patient)
            # scan, mask should be up shape: (scan length, height, width)
            scan = sitk.GetArrayFromImage(s)
            mask = sitk.GetArrayFromImage(m)
            unique, counts = np.unique(mask, return_counts=True)
            print('Saving patient dataset: ' + patient)
            data_export_MR_3D(scan, mask, FLAGS.saveDir, p_num, FLAGS.datasetName)
            p_num = p_num + 1
        create_tfrecord(os.path.join(FLAGS.saveDir, FLAGS.datasetName))
    else:
        patient_sets = find('mask_total*', data_path)
        patient_sets.sort()
        patient_sets = []
        for patient in patient_sets:
            print(patient)
            s = h5py.File(patient.replace('mask_total', 'scan'), 'r')
            m = h5py.File(patient, 'r')
            scan = s['scan'][:]
            mask = m['mask_total'][:]
            unique, counts = np.unique(mask, return_counts=True)
            print(unique)
            if (len(unique) >= 1) and (p_num >= 0):
                data_export_MR_3D(scan, mask, FLAGS.saveDir, p_num, FLAGS.datasetName, 7)
                print(p_num)
                p_num = p_num + 1
            else:
                incomplete.append(p_num)
                p_num = p_num + 1
        create_tfrecord(os.path.join(FLAGS.saveDir, FLAGS.datasetName))


if __name__ == '__main__':
  tf.app.run()

