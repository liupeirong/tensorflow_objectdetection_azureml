import base64
import json
import os
import time
import yaml
from io import BytesIO
import logging
import logging.config

import numpy as np
import tensorflow as tf
from azureml.core.model import Model
from PIL import Image

from utils import label_map_util
from azureml.monitoring import ModelDataCollector
from azure.storage.blob import BlockBlobService

MODEL_NAME = '__REPLACE_MODEL_NAME__'
PASCAL_LABEL_MAP_FILE = 'pascal_label_map.pbtxt'
THRESHOLD = float(os.getenv('MIN_CONFIDENCE', '0.8'))
IMAGE_STORAGE_ACCOUNT_NAME = '__REPLACE_IMAGE_STORAGE_ACCOUNT_NAME__'
IMAGE_STORAGE_ACCOUNT_KEY = '__REPLACE_IMAGE_STORAGE_ACCOUNT_KEY__'
IMAGE_STORAGE_CONTAINER_NAME = '__REPLACE_IMAGE_STORAGE_CONTAINER_NAME__'

def init():
  global logger
  global model
  global pred_collector
  global blob_service
  init_logger()
  pred_collector = ModelDataCollector(MODEL_NAME, identifier="imgpred", feature_names=["detection"])
  model = load_model()
  blob_service = BlockBlobService(IMAGE_STORAGE_ACCOUNT_NAME, IMAGE_STORAGE_ACCOUNT_KEY)
  blob_service.create_container(IMAGE_STORAGE_CONTAINER_NAME) #fail_on_exist=False by default

def init_logger():
  global logger
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
  logger = logging.getLogger('score')


def load_model():
  # figure out how many classes
  label_map_dict = label_map_util.get_label_map_dict(PASCAL_LABEL_MAP_FILE)
  num_classes = len(label_map_dict)
  logger.info('num_of_classes in pascal_label_map: {}'.format(num_classes))

  # Load label map
  label_map = label_map_util.load_labelmap(PASCAL_LABEL_MAP_FILE)
  categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # Load a frozen Tensorflow model into memory
  model_path = Model.get_model_path(MODEL_NAME)
  logger.info('getting model {} from {}'.format(MODEL_NAME, model_path))

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs }
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      sess = tf.Session(graph=detection_graph)
  return {
    'session': sess,
    'image_tensor': image_tensor,
    'tensor_dict': tensor_dict,
    'category_index': category_index
  }


def run(raw_data):
  logger.debug('raw_data: {}'.format(raw_data))
  try:
    response = inference(raw_data)
  except Exception as e:
    response = str(e)
  return response


def inference(raw_data):
  logger.info('parse input json raw_data to get image')
  parsed_json = json.loads(raw_data)
  image_raw = parsed_json['file']
  logger.info('base64 decode input image')
  image_bytes = base64.b64decode(image_raw)
  image = Image.open(BytesIO(image_bytes))
  logger.info('turn decoded image to np_array')
  (im_width, im_height) = image.size
  image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  image_np_expanded = np.expand_dims(image_np, axis=0)

  # Run inference
  start_time = time.time()
  output_dict = model['session'].run(
    model['tensor_dict'], feed_dict={model['image_tensor']: image_np_expanded})
  latency = time.time() - start_time
  logger.info('scoring took {} seconds'.format(latency))

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8).tolist()
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0].tolist()
  output_dict['detection_scores'] = output_dict['detection_scores'][0].tolist()
  logger.info('num of detections: {}'.format(output_dict['num_detections']))

  result = []
  for idx, score in enumerate(output_dict['detection_scores']):
    if score > THRESHOLD:
      result.append({
        'class': output_dict['detection_classes'][idx],
        'label': model['category_index'][output_dict['detection_classes'][idx]]['name'],
        'confidence': output_dict['detection_scores'][idx],
        'bounding_box': output_dict['detection_boxes'][idx]
      })
    else:
      logger.debug('idx {} detection score too low {}'.format(idx, score))
  #store the input image in blob storage and get the path to the image as correlation_id for prediction
  image_id = '{}/{}.jpg'.format(MODEL_NAME, int(start_time))
  StoreImage(image_id, image_bytes)
  pred_collector.collect(result, user_correlation_id=image_id)
  return result

def StoreImage(image_id, image_bytes):
  blob_service.create_blob_from_bytes(IMAGE_STORAGE_CONTAINER_NAME, image_id, image_bytes, 
    content_settings=ContentSettings('image/jpeg'))
