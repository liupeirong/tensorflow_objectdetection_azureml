# Automate the training and deployment process for Tensorflow Object Detection with Azure Machine Learning Service

[Tensorflow Object Detection](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md) is a powerful framework for creating computer vision models that can identify multiple objects in an image. This example demonstrates how Azure Machine Learning Service, and the [pipelines in Azure DevOps](https://dev.azure.com/paigedevops/machine-learning/_release?view=all&path=%5Ctensorflow-object-detection), can make it easy to train and deploy custom object detection models using Tensorflow Object Detection. Some of the capabilities this example enables include: 
* It supports creating a Docker image that contains all the dependencies required for training, including matching the Tensorflow version with the Object Detection code in GitHub, as well as matching dependencies for cpu or gpu compute resources that you choose. 
* You only need to point the training pipeline to the Azure Blob Storage location that contains your data and annotations. You can also retrain by simply adding additional images to storage. 
* Tensorflow training checkpoints are automatically stored in Azure Blob Storage and can be downloaded locally to view in Tensorboard.
* You only need to point the deployment pipeline to the version of the trained model you want to deploy, and it will deploy to an Azure Kubernetes cluster as a web service with security and monitoring automatically enabled. 

![Alt text](pictures/3_pipelines.png?raw=true "devops pipelines")

Below are detailed instructions of how to train a model, register a trained model, and deploy a registered model. 

## Pre-requisites:
1. Use a tool such as [VOTT](https://github.com/Microsoft/VoTT) to label your images, and export the results in Pascal VOC format.
2. Provision an [Azure Machine Learning Service Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace). You can do so in the Azure portal. Once provisioned, it will have the following resources associated with it - 
  * An Azure Blob Storage account for storing ML service metadata. You can also use it to store image data.
  * An Azure Container Registry that will store Docker images for web services built with trained models. You can also use it store custom Docker images.
  * An Azure App Insight account for monitoring deployed service.
3. Go to ML Service and provision -
  * An Azure ML compute compute resource for training. For Tensorflow object detection, a gpu-enabled compute resource will make training a lot faster.
  * An Azure Kubernetes Service for inference. This can be cpu based. But it will need to have a minimum of 3 nodes and a total minimum of 12 cores.   
4. [Create a Azure DevOps project](https://docs.microsoft.com/en-us/azure/devops/organizations/projects/create-project?view=azure-devops) if you don't already have one, so that we can later release pipelines for training and deployment.

## Train a model
### Prepare data
1. Use the Dockerfile or Dockerfile_gpu to build a Docker image that contains all the dependencies for training. Register the image with the Azure Container Registry associated with the Workspace. Azure Machine Learning Service supports using custom Docker images for training. 
```bash
docker build -f Dockerfile_gpu ./
docker tag your_image_id your_acr.azurecr.io/tensorflowobjectdetection:1
docker push your_acr.azurecr.io/tensorflowobjectdetection:1
```
2. The ML Workspace has a concept of [datastores](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-azure-machine-learning-architecture#datastore). You can put everything in the default datastore, or you can register additional datastores for your project data and for Tensorflow training checkpoints. See an example of how to do this in the [aml-train Jupyter notebook](aml_train/aml-train.ipynb).
3. To train a model, we need: 
  *  labeled image data in Pascal VOC format
  *  a base model such as faster_rcnn or ssd_mobilenet models - you can download a base model from the [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). However, the older configuration in the Model Zoo may need to be adjusted to match the newer Tensorflow code. For example, see [this issue](https://github.com/tensorflow/models/issues/3794) to adjust the pipeline.config in a faster_rcnn model after you download  the model from the Model Zoo. 
4. Copy data to your target datastore (Azure Blob Storage) such that it lands in a folder structure described at the beginning of the [aml-train Jupyter notebook](aml_train/aml-train.ipynb). You can use [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?toc=%2fazure%2fstorage%2fblobs%2ftoc.json) to efficiently copy data to Azure Blob Storage. For example, organize your data locally in the following folder structure:

![Alt text](pictures/2_localdatadir.png?raw=true "folder structure")

Then copy to Blob Storage using azcopy:
```bash
azcopy --source /pets --destination https://your_account.blob.core.windows.net/your_contain/pets --dest-key your_key --recursive
```    

### Train from Jupyter Notebook
1. Set environment variables for your ML workspace:
```bash
export SUBSCRIPTION_ID="azure_subscription_id"
export RESOURCE_GROUP="resource_group_for_azure_ml_workspace"
export WORKSPACE_NAME="azure_ml_workspace_name"
export PROJ_DATASTORE="azure_ml_datastore_name_where_data_and_base_model_are_stored"
export LOGS_DATASTORE="azure_ml_dstastore_name_where_tensorflow_checkpoints_will_be_stored"
export ACR_ID="azure_container_registry_account.azurecr.io"
export ACR_USERNAME="azure_container_registry_account"
export ACR_PASSWORD="azure_container_registry_password"
export AML_COMPUTE_GPU="azure_ml_gpu_compute_name"
export TRAINING_DOCKER_SHORT_NAME="docker_image_name:image_version"
```
2. Start Jupyter notebook in the context of the environment variables set above.
3. Open [aml-train.ipynb](aml_train/aml-train.ipynb), set the training parameters and run the notebook. 
![Alt text](pictures/1_trainvars.png?raw=true "training variables")
Note that by default, Pascal VOC format has the class name in the annotation xml, however, if you use the Oxford-IIIT Pets dataset, the class name is in the image file name. So you just need to set the variable classname_in_filename to True.

### Train from Azure DevOps pipeline
1. Create a variable group in your Azure DevOps project to set the environment variables for your ML workspace. See an example [here](https://dev.azure.com/paigedevops/machine-learning/_releaseProgress?releaseId=38&_a=release-variables).
2. Create a [training pipeline](https://dev.azure.com/paigedevops/machine-learning/_release?view=all&_a=releases&definitionId=6) to submit training to Azure ML. 
3. Set the variables for the training pipeline. See an example [here](https://dev.azure.com/paigedevops/machine-learning/_releaseProgress?releaseId=40&_a=release-variables).
4. Create a release which will automatically start an experiment run in Azure ML by running the training Docker container in the target Azure ML compute.  The code will  
  *  automtically update the base model's pipeline.config with 
     *  the number of classes based on pascal_label_map.pbtxt in the annotation
     *  the number of training steps specified in the release pipeline variable
     *  checkpoint path and label_map path in pipeline.config
  *  create tfrecords from annotations. 
  *  upload the training checkpoint data to the target datastore for logs.  

## Register a model
1. Download the training checkpoints from Azure Blob Storage to evaluate the model in Tensorboard locally:
```bash
azcopy --source https://your_account.blob.core.windows.net/your_log_container/your_proj_root/yyyymmdd_HHMM/model --source-key your_key --destination ./tflogs --recursive
# you may see some permission errors on accessing subfolders, but they can be ignored.
tensorflog --logdir=./tflogs
```
2. If you are happy with the model, register it with Azure ML by running the [register model pipeline](https://dev.azure.com/paigedevops/machine-learning/_release?view=all&_a=releases&definitionId=7):
  *  Set the pipeline variables, see example [here](https://dev.azure.com/paigedevops/machine-learning/_releaseProgress?releaseId=38&_a=release-variables). ```RUN_ID``` is the Azure ML experiment run id that produced the model you want to register.  You can get it programmatically or from the Azure portal as shown below.

    ![Alt text](pictures/4_runid.png?raw=true "run_id")
    
  *  The sample pipeline has human approval enabled, you can set the variable before approving.

## Deploy to Azure Kubernetes Service
1. Run the [deployment pipeline](https://dev.azure.com/paigedevops/machine-learning/_release?view=all&definitionId=8) to deploy a registered model as a web service to Azure Kubernetes Service.
  *  Set the pipeline variables, see example [here](https://dev.azure.com/paigedevops/machine-learning/_releaseProgress?releaseId=39&_a=release-variables). ```MODEL_VERSION``` is the version of the registered model in Azure ML you want to deploy.  You can get it programmatially or from the Azure portal. 
2. Azure ML will automatically create a web service that features the following:
  *  Highly available by leveraging Kubernetes
  *  Secured by an API key 
  *  Performance metrics are sent to App Insights for monitoring
  *  Input data and inference results are [collected in Azure Blob Storage](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-data-collection). In this example we are only generating a unique input image ID and store it with the inference results.
3. To consume the web service, see an example at the end of the [aml-deploy Jupyter notebook](aml_deploy/aml-deploy.ipynb). 

