# Scalable ML Lab 1
## Alexander Yonchev
## Fernando Vallecillos Ruiz

X = {iris, titanic}

#### X-feature-pipeline.py
This script reads the dataset and create creates a feature group in hopsworks. It will then be accesible online for its posterior use.

#### X-feature-pipeline-daily.py
This script checks if there is a feature group already created in hopsworks, if not it will behave as the previous one. Otherwise, it will generate a random data sample and add it to the feature group.

#### X-training-pipeline.py
This script access the dataset through hopsworks feature group. It splits splits it into training and testing and creates a K-nearest-neighbor model with the training data. It uses the testing set to create a confusion matrix and saves it as an image along with the pickled classifier. It creates a entry in hopsworks to save the schema of the model along with some of its metrics.


#### X-batch-inference-pipeline.py
This script access the saved model in hopsworks to make new predictions. Instead of generating a new flower, it reads the dataset and uses the model to predict the whole dataset. It saves an image of the proper answer and the predicted one. It will continue by downloading (or creating) a feature group about the history of the predictions to be able to monitor it later on. If it has predicted at least a member of each of the classes, it will create a confusion matrix and save it as an image.

***

Both projects have online interfaces to generate predictions and monitor the history:
## Iris
https://huggingface.co/spaces/nandovallec/iris
https://huggingface.co/spaces/nandovallec/iris-monitor
## Titanic
https://huggingface.co/spaces/nandovallec/titanic
https://huggingface.co/spaces/nandovallec/titanic-monitor

---
The notebook TitanicData has been added show a possible way to clean the original dataset. 

