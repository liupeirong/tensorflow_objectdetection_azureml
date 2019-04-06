# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    in the research folder run:
    python path/to/create_tf_record.py \
        --data_dir=path/to/root_of_subfolders_of_PascalVoC \
        --output_dir=path/to/generated_tfrecords_for_all_subfolders
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
import re

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import random
import csv
import filecmp
import logging
import logging.config
import yaml
import shutil

from os import listdir
from os.path import isfile, join

flags = tf.app.flags
flags.DEFINE_string('data_dir', None, 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_dir', None, 'Path to output TFRecord')
flags.DEFINE_string('label_map_file', 'pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
flags.DEFINE_boolean('force_regenerate', False, 'If true will regenerate TFrecords '
                     'even if the existing TFrecords were created after the last modification of all the sub folders.')
flags.DEFINE_string('log_dir', 'logs', 'Folder where the log files should go.')
flags.DEFINE_boolean('classname_in_filename', False, 'Whether classname is in filename')
FLAGS = flags.FLAGS

RANDOM_SEED = 43
logger = None

def data_changed(root_data_dir, train_output_file):
  #if no output yet
  if not os.path.isfile(train_output_file):
    return True

  #if output is older than data files
  lastmodtime = max(os.stat(root).st_mtime for root,_,_ in os.walk(root_data_dir))  
  outputtime = os.stat(train_output_file).st_mtime
  logger.info('data folders last updated {}, training tf records last genereated {}'.format(lastmodtime, outputtime))
  return lastmodtime > outputtime 


def get_label_map(root_data_dir):
  first = True
  logger.info('get_label_map root_data_dir {}'.format(root_data_dir))
  dirs = os.listdir(root_data_dir)
  for subdir in dirs:
    if first:
      sample_file = os.path.join(root_data_dir, subdir, FLAGS.label_map_file)
      logger.info('base pascal label file: {}'.format(sample_file))
      first = False
      label_map_dict = label_map_util.get_label_map_dict(sample_file)
    else:
      if not filecmp.cmp(sample_file, os.path.join(root_data_dir, subdir, FLAGS.label_map_file), shallow=False):
        logger.error('label map file in {} is different from {}'.format(subdir, sample_file))
        label_map_dict = None
        break
  return dirs, label_map_dict, sample_file


def split_train_val_test(data_dir, 
                         annotations_dir):
  logger.info('spliting in {}'.format(annotations_dir))
  allxml = [f for f in listdir(annotations_dir) if isfile(join(annotations_dir, f))]
  random.Random(RANDOM_SEED).shuffle(allxml)
  cnt = len(allxml)
  train = allxml[:int(cnt*.70)]
  val = allxml[int(cnt*.70):int(cnt*.90)]
  test = allxml[int(cnt*.90):]

  with open(os.path.join(data_dir, 'train.txt'), 'w') as resultFile:
    for x in train:
      resultFile.write(x + "\n")

  with open(os.path.join(data_dir, 'val.txt'), 'w') as resultFile:
    for x in val:
      resultFile.write(x + "\n")

  with open(os.path.join(data_dir, 'test.txt'), 'w') as resultFile:
    for x in test:
      resultFile.write(x + "\n")

  return train, val, test


def generate_tf_for_set(set_file_list,
                        set_writer,
                        data_dir,
                        annotations_dir,
                        label_map_dict):
  for idx, set_file in enumerate(set_file_list):
    if idx % 5 == 0:
      logger.debug('On image %d of %d', idx, len(set_file_list))
    path = os.path.join(annotations_dir, set_file)
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, data_dir, label_map_dict, FLAGS.ignore_difficult_instances)
    set_writer.write(tf_example.SerializeToString())


def process_one_subdir(data_dir,
                       label_map_dict,
                       train_writer,
                       val_writer,
                       test_writer):
  annotations_dir = os.path.join(data_dir, 'Annotations')
  train, val, test = split_train_val_test(data_dir, annotations_dir)
  logger.info('generate training tfrecords')
  generate_tf_for_set(train, train_writer, data_dir, annotations_dir, label_map_dict)
  logger.info('generate validation tfrecords')
  generate_tf_for_set(val, val_writer, data_dir, annotations_dir, label_map_dict)
  logger.info('generate testing tfrecords')
  generate_tf_for_set(test, test_writer, data_dir, annotations_dir, label_map_dict)

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.
  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"
  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  if not os.path.isfile(full_path):
    if os.path.isfile(full_path + '.JPG'):
      full_path = full_path + '.JPG'
    elif os.path.isfile(full_path + '.jpg'):
      full_path = full_path + '.jpg'    
  logger.info('process image {}'.format(full_path))
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  logger.info('here again {}'.format(FLAGS.classname_in_filename))
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      if FLAGS.classname_in_filename:
        class_name = get_class_name_from_filename(data['filename'])
      else:
        class_name = obj['name']
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  global logger
  os.makedirs(FLAGS.log_dir, exist_ok=True)
  logconf = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logging.yml')
  print('logconf {}'.format(logconf))
  if os.path.exists(logconf):
    with open(logconf, 'rt') as f:
      config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    print('logconf loaded')
  else:
    logging.basicConfig(level=default_level) 
    print('logconf fall back to default')
  logger = logging.getLogger('createTFRecord')

  root_data_dir = FLAGS.data_dir
  output_dir = FLAGS.output_dir
  logger.info('data_dir: {}'.format(root_data_dir))
  logger.info('output_dir: {}'.format(root_data_dir))
  logger.info('classname is in filename: {}'.format(FLAGS.classname_in_filename))
  os.makedirs(output_dir, exist_ok=True)

  train_output_file = os.path.join(output_dir, 'train.record')
  val_output_file = os.path.join(output_dir, 'val.record')
  test_output_file = os.path.join(output_dir, 'test.record')

  #if tfrecords were already generated after all the subdirectories were created,do nothing
  if not FLAGS.force_regenerate and not data_changed(root_data_dir, train_output_file):
    logger.info('data directories have not changed...exiting')
    return 

  #check if all subfolders have the same shape of data
  dirs, label_map_dict, label_map_file = get_label_map(root_data_dir)
  if label_map_dict is None:
    logger.error('label map file must be same in all sub folders...exiting')
    return 

  train_writer = tf.python_io.TFRecordWriter(train_output_file)
  val_writer = tf.python_io.TFRecordWriter(val_output_file)
  test_writer = tf.python_io.TFRecordWriter(test_output_file)
  
  for subdir in dirs:
    process_one_subdir(os.path.join(root_data_dir, subdir), label_map_dict, train_writer, val_writer, test_writer) 

  train_writer.close()
  val_writer.close()
  test_writer.close()
  
  shutil.copy(label_map_file, output_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('data_dir')
  flags.mark_flag_as_required('output_dir')
  tf.app.run()
