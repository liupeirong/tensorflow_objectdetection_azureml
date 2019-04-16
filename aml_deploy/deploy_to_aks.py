import string
import os
import argparse
import shutil

import azureml.core
from azureml.core import Workspace, Datastore
from azureml.core.model import Model
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import ContainerImage, Image

print("SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--proj_root', 
                    help='root directory in Azure ML data store that contains training data and base model',
                    required=True)
parser.add_argument('--model_version', type=int,
                    help='the version of the model to deploy',
                    required=True)
parser.add_argument('--support_gpu', type=bool, 
                    help='Whether to inference with gpu-enabled compute',
                    default=False)
FLAGS = parser.parse_args()

# input parameters
proj_root=FLAGS.proj_root
model_version = FLAGS.model_version
support_gpu = FLAGS.support_gpu

# environment variables
subscription_id = os.getenv('SUBSCRIPTION_ID')
resource_group = os.getenv('RESOURCE_GROUP')
workspace_name = os.getenv('WORKSPACE_NAME')
proj_datastore = os.getenv('PROJ_DATASTORE', default=None)
compute_cpu = os.getenv('AML_AKS_CPU', default='akscpu')
compute_gpu = os.getenv('AML_AKS_GPU', default='aksnv6')
image_storage_account_name = os.getenv('IMAGE_STORAGE_ACCOUNT_NAME')
image_storage_account_key = os.getenv('IMAGE_STORAGE_ACCOUNT_KEY')
image_storage_container_name = os.getenv('IMAGE_STORAGE_CONTAINER_NAME')

# constants
DATA_SUBDIR='data'
TFRECORDS_SUBDIR='tfrecords'
MODELS_SUBDIR='models'
PASCAL_LABEL_MAP_FILE='pascal_label_map.pbtxt'

# set up Azure ML environment
ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
if proj_datastore is None:
    ds = ws.get_default_datastore()
else:
    ds = Datastore.get(ws, datastore_name=proj_datastore)
compute_name = compute_gpu if support_gpu else compute_cpu

#   derive model_name from proj_root, same logic as in training
model_name = proj_root if proj_root.isalnum() else ''.join(ch for ch in proj_root if ch.isalnum())
model = None
models = Model.list(ws, name=model_name)
for m in models:
    if m.version == model_version:
        model = m
        break
if model is None:
    raise ValueError('model {}:{} not found'.format(model_name, model_version))
print("proj_root:{}, model:{}:{}".format(proj_root, model_name, model_version))

pascal_label_map_in_ds = os.path.join(proj_root, TFRECORDS_SUBDIR, PASCAL_LABEL_MAP_FILE)
inference_docker_image = model_name
webservice_name = model_name + 'svc'

# Build the inference image
#   inference depends on 2 files:
#       1) the frozen model 
#       2) the label map so we can display human-readble labels 
ds.download(target_path='.',
            prefix=pascal_label_map_in_ds,
            overwrite=True,
            show_progress=True) 
shutil.copy(pascal_label_map_in_ds, '.') #copy to current folder

#   score.py has to load the model, but there's no way to pass in a parameter or set env variable to the image or web service
with open("score.py", "rt") as fin:
    with open("mscore.py", "wt") as fout:
        for line in fin:
            fout.write(line
            .replace('__REPLACE_MODEL_NAME__', model_name)
            .replace('__REPLACE_IMAGE_STORAGE_ACCOUNT_NAME__', image_storage_account_name)
            .replace('__REPLACE_IMAGE_STORAGE_ACCOUNT_KEY__', image_storage_account_key)
            .replace('__REPLACE_IMAGE_STORAGE_CONTAINER_NAME__', image_storage_container_name))

image_config = ContainerImage.image_configuration(
    execution_script = "mscore.py",
    runtime = "python",
    conda_file = "conda_env.yml",
    description = model_name,
    dependencies = ['./', 'utils/'],
    enable_gpu = support_gpu)

image = ContainerImage.create(name = inference_docker_image, 
                              models = [model], 
                              image_config = image_config,
                              workspace = ws
                              )
image.wait_for_creation(show_output=True)
inference_docker_image_version = image.version
print('created image: {}:{}'.format(inference_docker_image, inference_docker_image_version))

# Deploy a new web service or update an existing web service with the image
def deploy_new_webservice(ws, compute_name, webservice_name, image):
    aks_target = ws.compute_targets[compute_name]
    aks_config = AksWebservice.deploy_configuration(collect_model_data=True, enable_app_insights=True)
    service = Webservice.deploy_from_image(
        workspace = ws, 
        name = webservice_name,
        image = image,
        deployment_config = aks_config,
        deployment_target = aks_target)
    service.wait_for_deployment(show_output = True)
    print(service.state)

def update_existing_webservice(service, image):
    service.update(image = image)
    service.wait_for_deployment(show_output = True)
    print(service.state)

# Check if the target service already exists
try: 
    service = Webservice(name = webservice_name, workspace = ws)
except:
    deploy_new_webservice(ws, compute_name, webservice_name, image)
else:
    update_existing_webservice(service, image)

