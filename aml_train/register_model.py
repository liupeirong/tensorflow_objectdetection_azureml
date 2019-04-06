import os
import argparse
import azureml.core
from azureml.core import Workspace, Experiment, Run

print("SDK version:", azureml.core.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--proj_root', 
                    help='root directory in Azure ML data store that contains training data and base model',
                    required=True)
parser.add_argument('--run_id', 
                    help='the experiment run_id (not the run number) in Azure ML that the model is based on',
                    required=True)
FLAGS = parser.parse_args()

# input parameters
proj_root=FLAGS.proj_root
run_id = FLAGS.run_id

# env variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")

# constants
MODEL_FILE='outputs/model/frozen_inference_graph.pb'

# set up Azure ML environment
ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
model_name = proj_root if proj_root.isalnum() else ''.join(ch for ch in proj_root if ch.isalnum())
experiment_name = model_name
exp = Experiment(workspace=ws, name=experiment_name)

print("experiment:{}".format(experiment_name))

# register the model
run = Run(exp, run_id)
model = run.register_model(model_name=model_name, model_path=MODEL_FILE)
print('registered model {}, version: {}'.format(model.name, model.version))

