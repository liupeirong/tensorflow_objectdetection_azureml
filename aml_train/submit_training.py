import string
import os
import argparse
import datetime
import azureml.core
from azureml.core import Workspace, Experiment, Run, Datastore, ScriptRunConfig
from azureml.core.runconfig import ContainerRegistry, RunConfiguration, DataReferenceConfiguration

print("SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--proj_root', 
                    help='root directory in Azure ML data store that contains training data and base model',
                    required=True)
parser.add_argument('--base_model_dir', 
                    help='Path to a downloaded base model which contains model.ckpt and pipeline.config.',
                    required=True)
parser.add_argument('--num_steps', type=int, 
                    help='Number of train steps.',
                    default=20000)
parser.add_argument('--force_regenerate_tfrecords', type=bool, 
                    help='Whether to regenerate TFRecords even if no changes detected',
                    default=False)
parser.add_argument('--support_gpu', type=bool, 
                    help='Whether to train with gpu-enabled compute',
                    default=True)
parser.add_argument('--classname_in_filename', type=bool, 
                    help='Whether class name is in filename instead of annotation xml',
                    default=False)
FLAGS = parser.parse_args()

# input parameters
proj_root=FLAGS.proj_root
base_model = FLAGS.base_model_dir
force_regenerate_tfrecords = FLAGS.force_regenerate_tfrecords
training_steps = FLAGS.num_steps
support_gpu = FLAGS.support_gpu
classname_in_filename = FLAGS.classname_in_filename

# env variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
proj_datastore = os.getenv("PROJ_DATASTORE", default = None)
logs_datastore = os.getenv("LOGS_DATASTORE", default = 'logsds')
docker_registry_address = os.getenv("ACR_ID")
docker_registry_username = os.getenv("ACR_USERNAME")
docker_registry_password = os.getenv("ACR_PASSWORD")
compute_cpu = os.getenv("AML_COMPUTE_CPU", default = 'amlcpu')
compute_gpu = os.getenv("AML_COMPUTE_GPU", default = 'amlnv6')
training_docker_image_short_name = os.getenv("TRAINING_DOCKER_SHORT_NAME", default='tensorflowobjectdetection:1')

# constants
DATA_SUBDIR='data'
TFRECORDS_SUBDIR='tfrecords'
MODELS_SUBDIR='models'
SCRIPT_FOLDER = './scripts'
SCRIPT_FILE = 'train.py'

# set up Azure ML environment
ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
if proj_datastore is None:
    ds = ws.get_default_datastore()
else:
    ds = Datastore.get(ws, datastore_name=proj_datastore)
dslogs = Datastore.get(ws, datastore_name=logs_datastore)
print(ds.container_name, dslogs.container_name)

compute_name = compute_gpu if support_gpu else compute_cpu
compute_target = ws.compute_targets[compute_name]
model_name = proj_root if proj_root.isalnum() else ''.join(ch for ch in proj_root if ch.isalnum())
experiment_name = model_name
exp = Experiment(workspace=ws, name=experiment_name)

print("datastore:{}, compute:{}".format(ds.container_name, type(compute_target)))
print("proj_root:{}, model_name:{}".format(proj_root, model_name))

image_registry_details = ContainerRegistry()
image_registry_details.address = docker_registry_address
image_registry_details.username = docker_registry_username
image_registry_details.password = docker_registry_password
training_docker_image = docker_registry_address + '/' + training_docker_image_short_name

# set up training configuration
dr = DataReferenceConfiguration(datastore_name=ds.name, 
                                path_on_datastore=proj_root,
                                overwrite=True)
drlogs = DataReferenceConfiguration(datastore_name=dslogs.name, 
                                path_on_datastore=proj_root,
                                overwrite=True)
run_cfg = RunConfiguration()
run_cfg.environment.docker.enabled = True
run_cfg.environment.docker.gpu_support = support_gpu
run_cfg.environment.docker.base_image = training_docker_image 
run_cfg.environment.docker.base_image_registry = image_registry_details
run_cfg.data_references = {ds.name: dr, dslogs.name: drlogs} 
run_cfg.environment.python.user_managed_dependencies = True
run_cfg.target = compute_target

# submit training
currentDT = datetime.datetime.now()
currentDTstr = currentDT.strftime("%Y%m%d_%H%M")
print('logs will be in the logs data store in folder {}'.format(currentDTstr))

base_mount = ds.as_mount() 
data_dir = os.path.join(str(base_mount), DATA_SUBDIR)
tfrecords_dir = os.path.join(str(base_mount), TFRECORDS_SUBDIR)
base_model_dir = os.path.join(str(base_mount), MODELS_SUBDIR, base_model)
logs_mount = dslogs.as_mount()
logs_dir = os.path.join(str(logs_mount), currentDTstr)

script_params = [
    '--data_dir', data_dir,
    '--base_model_dir', base_model_dir, 
    '--tfrecords_dir', tfrecords_dir,
    '--force_regenerate_tfrecords', force_regenerate_tfrecords,
    '--num_steps', training_steps,
    '--log_dir', logs_dir,
    '--classname_in_filename', classname_in_filename
]

src = ScriptRunConfig(source_directory = SCRIPT_FOLDER, 
                      script = SCRIPT_FILE, 
                      run_config = run_cfg,
                      arguments=script_params)

run = exp.submit(src)
print('run details {}'.format(run.get_details))
