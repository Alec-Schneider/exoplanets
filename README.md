# Exoplanet Hunting in Deep Space on Amazon Sagemaker

Using the Exoplanet dataset via [Kaggle](https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data), I built a binary classification neural network in [PyTorch](https://pytorch.org/) to find exoplanets based on the change in flux (light intensity) of stars.

![wavelengths](./data/no_exoplanets_wavelengths.png)

## The Dataset

[EDA Notebook](./eda_exoPlanet.ipynb)

The training dataset consists of a 5087x3198 matrix, with the label column containing values of 2, a star containing one or more exoplanets in orbit, and 1, a star with no exoplanets in orbit. The test is a 570x3198 matrix containing the same label columns. In order to get our predictions to be in line with the predicted probabilities of a class (between 0-1) the labels need to be adjusted to 1 and 0:

```python
orig_train = pd.read_csv("data/exoTrain.csv")
orig_train['LABEL'] = orig_train['LABEL'] - 1
orig_train.to_csv("data/Train.csv", index=False)

orig_test = pd.read_csv("data/exoTest.csv")
orig_test['LABEL'] = orig_test['LABEL'] - 1
orig_test.to_csv("data/Test.csv", index=False)
```

### Visualize the data via Principal Component Analysis

[Interactive 3D Plot of the data transofrmed by PCA ](https://htmlpreview.github.io/?https://github.com/Alec-Schneider/exoplanets/blob/main/data/pca3_plot.html)

#### Scatter Plot of the PCA Transformed data

![scatter](./data/pca2_scatter.png)

#### Zoomed in view of the data by label

![zoomed](./data/pca2_zoomed.png)

To first prepare the data for training [Sklearn's StandardScaler]() will scale the data so the neural network will learn faster, as the values in the dataset are significantly large. Next [Scipy's uniform_filter1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter1d.html) was applied to the scaled data to create a 200 point moving average across the array for each star in the training and testing set. This will also add another dimension to our dataset, giving it a second channel. With the data prepared for training, a Dataset class can be crafted to handle the data for a PyTorch Dataloader.

As we can see from the plots in the visualization section, the dataset is heavily imbalanced with 5050, or 99.27%, of the training data consisting of label=0. To alleviate this during training, we can rebalance the dataset by creating new samples of exoplanets data points by rotating the data points using [numpy's roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) function on a sample array. In the ExoplanetDataset class below, a [PyTorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) is created from the scaled training and test data. If a training dataset is being created, then additional samples belonging to class "1" are created and added to training dataset. Now to ease the process of creating a [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), which is need to pass to a PyTorch model, the get_data_loader function was created to return the DataLoader with a specified batch size (e.g. 16, 32, 64) and shuffle the data to ensure the model does not memorize the data as it is during training.

```python
class ExoplanetDataset(Dataset):
    """
    Dataset PyTorch class to retrieve the training and test data of the Exoplanents dataset
    """

    def  __init__(self, train=True):
        self.root = "./data"
        self.train = train
        self.x_train_file = "x_train_scaled_filt.txt"
        self.y_train_file = "y_train.csv"
        self.x_test_file = "x_test_scaled_filt.txt"
        self.y_test_file = "y_test.csv"

        if self.train:
            print("Loading Training Data")
            self.x_train = self.load_features(self.x_train_file)
            self.y_train = pd.read_csv(os.path.join(self.root, self.y_train_file))[['LABEL']]


            pos_idx = np.where(self.y_train == 1)[0]
            neg_idx = np.where(self.y_train == 0)[0]
            x_pos = self.x_train[pos_idx]
            x_neg = self.x_train[neg_idx]
            # Add rotations for the positives to the dataset to upsample
            num_rotations = 100
            for i in range(len(x_pos)):
                 for r in range(num_rotations):
                        rotated_row = np.roll(x_pos[i,:], shift = r, axis=0)
                        self.x_train = np.vstack([self.x_train, rotated_row[np.newaxis, :, :]])

            self.y_train = np.vstack([self.y_train,
                                      np.array([1] * len(x_pos) * num_rotations).reshape(-1,1)])\
                            .reshape(-1)
            self.len = len(self.y_train)

        else:
            self.x_test = self.load_features(self.x_test_file)
            self.y_test = pd.read_csv(os.path.join(self.root, self.y_test_file))['LABEL'].values
            self.len = len(self.y_test)

    def __getitem__(self, idx):
        if self.train:
            features = torch.tensor(self.x_train[idx])
            labels = torch.tensor(self.y_train[idx])

        else:
            features = torch.tensor(self.x_test[idx])
            labels = torch.tensor(self.y_test[idx])

        return (features, labels)

    def __len__(self):
        return self.len

    def load_features(self, file):
        loaded_arr = np.loadtxt(os.path.join(self.root, file))
        features = loaded_arr.reshape(
            loaded_arr.shape[0], loaded_arr.shape[1] // 2, 2)

        return features

def get_data_loader(batch_size, train=True):
    """
    Helper function to create a data loader from the ExoplanetDataset
    """
    logger.info("Get data loader")

    if train:
        dataset = ExoplanetDataset(train=True)
    else:
        dataset = ExoplanetDataset(train=False)

    tensor_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return tensor_dataloader
```

## Training The Classifier

With the data processing and data prep for model training taken care of, a model architecture needs to be constructed. For the classifier, a CNN architecture will be used with the following design:

- Two [1-Dimensional Convolutional](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) layers
- Two [1-Dimensional Max Pooling](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool1d.html) layers
- Two [1-Dimensional BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) layers
- Three Linear layers using the CNN parameters as input to make a single binary prediction using the [sigmoid function](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html)

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 3)
        self.b_norm1 = nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.b_norm2 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16 * 354, 500)
        self.fc2 = nn.Linear(500, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Transform the input shape of the batch
        x = torch.unsqueeze(x, 2)
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = self.b_norm1(x)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = self.b_norm2(x)
        x = x.view(-1, 16 * 354)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
```

Now all that is needed to train and test the model is to write training and testing functions:

```python
def train_model(args):
    """
    args: NameSpace type with model training arguments

    return: Trained PyTorch Model
    """
    global losses

    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set the seed
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    try:
        train_loader = args.train_loader
    except:
        train_loader = get_data_loader(args.batch_size, train=True)

    try:
        test_loader = args.test_loader

    except:
        test_loader = get_data_loader(args.test_batch_size, train=False)

    model = args.model
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):

        total_loss = 0
        model.train()

        for step, batch in enumerate(train_loader):
            feats = batch[0].to(device)
            labels = batch[1].to(device)

            model.zero_grad()

            outputs = model(feats.float())

            loss = criterion(torch.squeeze(outputs, 1).float(), labels.float())
            total_loss += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
        logger.info("Average training loss: %f\n", total_loss / len(train_loader))
        test(model, test_loader, device)

        losses.append(total_loss)
    logger.info("Saving tuned model")

    model_2_save = model.module if hasattr(model, "module") else model
     # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, f'{args.model_name}.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

    return model


def test(model, test_loader, device):
    global accuracies

    def get_correct_count(preds, labels):
        pred_flat = np.round(preds,0).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat), len(labels_flat)

    model.eval()
    _, eval_accuracy = 0, 0
    total_correct = 0
    total_count = 0


    with torch.no_grad():
        for batch in test_loader:

            b_input_ids = batch[0].to(device)
            b_labels = batch[1].to(device)

            outputs = model(b_input_ids.float())
#             preds = outputs[0]
            preds = outputs.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()


            num_correct, num_count = get_correct_count(preds, label_ids)
            total_correct += num_correct
            total_count += num_count
    accuracy = total_correct/total_count
    accuracies.append(accuracy)
    logger.info("Test set: Accuracy: %f\n", accuracy)
```

Kicking off the training on the model involves creating a SimpleNamespace containing to model that is to be trained, the hyperparameters of the model, the DataLoaders we would like to use as training and testing, and paths for saving the trained model.

```python
accuracies = []
losses = []
cnn = CNN()
args = SimpleNamespace(
    batch_size=24,
    test_batch_size=16,
    epochs=10,
    lr=1e-5,
    seed=1,
    model=cnn,
    log_interval =50,
    model_dir = "model/",
    model_name="cnn_10epochs_24batch",
    data_dir="./data/",
    train_loader=train_loader,
    test_loader=test_loader,
    num_gpus=1,
    x_train_file = "x_train_scaled_filt.txt",
    y_train_file="y_train.csv",
    x_test_file="x_test_scaled_filt.txt",
    y_test_file="y_test.csv"
)

cnn_model = train_model(args)
```

With this CNN model design, 10 epochs, and a learning rate of 1e-5, 97% accuracy is achieved on the test set. More metrics including a ROC-AUC score of 55.9% and precision, recall, and f1-score metrics can be found in the training [notebook](./exoPlanet_classifier.ipynb).

## Training The Classifier With Amazon SageMaker

[Sagemaker Training Notebook](./exoplanets_sagemaker.ipynb)

The training step can also be acheived by running a training job on Amazon SageMaker via the [Python PyTorch SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html), which can later be used to host the model for others to pass info to the model for predictions.

To launch a training job there first needs to be a creation of a source code directory, which I called "code" and write a training script, called train.py. The training script includes the previously mentioned CNN class, the ExoplanentsDataset class, get_data_loader function, slightly modified versions of the train_model and test functions, plus some additional command line parameter handling specific to SageMaker.

Once the training script has been finalized, and an S3 bucket associated with the SageMaker app has been identified, the job can be executed with the below code.

```python
import os
import numpy as np
import pandas as pd
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = "sagemaker-studio-772149141904-nrs4q1pu91"
prefix = "sagemaker/pytorch-exoplanets"

role = sagemaker.get_execution_role()

x_inputs_train = sagemaker_session.upload_data("./data/data_train_x_scaled_filt.txt", bucket=bucket, key_prefix=prefix)
y_inputs_train = sagemaker_session.upload_data("./data/data_train_y.csv", bucket=bucket, key_prefix=prefix)

x_inputs_test = sagemaker_session.upload_data("./data/data_test_x_scaled_filt.txt", bucket=bucket, key_prefix=prefix)
y_inputs_test = sagemaker_session.upload_data("./data/data_test_y.csv", bucket=bucket, key_prefix=prefix)

output_path = f"s3://{bucket}/{prefix}"

from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="code",
    role=role,
    framework_version="1.6",
    py_version="py3",
    instance_count=1,
    instance_type="ml.g4dn.12xlarge",
    output_path=output_path,
    hyperparameters={
        "epochs": 10,
        "lr" : 1e-5,
        "batch-size" : 24,
        "test-batch-size" : 16,
        "log-interval": 50,
        "num-gpus": 1,
        "model-name": "torch_cnn"
    }

)

estimator.fit({"training": 's3://sagemaker-studio-772149141904-nrs4q1pu91/sagemaker/pytorch-exoplanets/data_',
               "testing": 's3://sagemaker-studio-772149141904-nrs4q1pu91/sagemaker/pytorch-exoplanets/test_'})
```

First, the execution role of the SageMaker API is fetched so it can be passed to the PyTorch SDK. Second, the training and testing data is uploaded to the S3 bucket. Lastly, the training job is kicked off by instantiating the SageMaker PyTorch estimator and calling the .fit() function. The total job, with 1 GPU, takes about 12 minutes to complete and return a trained PyTorch CNN model.

Using the fitted estimator, the model can be deployed for users to pass data to:

```python
from sagemaker.pytorch.model import PyTorchModel

model_data = estimator.model_data

pytorch_model = PyTorchModel(model_data=model_data,
                             role=role,
                             framework_version="1.6",
                             source_dir="code",
                             py_version="py3",
                             entry_point="inference.py")


predictor = pytorch_model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")

```

## Appendix

### First Dataset Imbalance Correction Method

At first I attempted to use the below sampling method to improve model performance, however it was not as good as expected. The method of creating rotations of existing exoplanet data was found to be a better method. See below:

To alleviate the imbalance in the dataset during training, we can rebalance the dataset by by passing a sampler to the [DataLoader](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader) that will be passed to our PyTorch model. An implementation of a Imbalanced dataset sampler exists [here](https://github.com/ufoym/imbalanced-dataset-sampler), so I forked it to make an edit for the custom [TensorDataset](https://pytorch.org/docs/stable/data.html?highlight=tensordataset#torch.utils.data.TensorDataset) that is being used in this project.

With a solution to the imbalance, I'll define a function to retrieve a DataLoader.

```python
def get_data_imbalanced_loader(batch_size, training_dir, filename):
    """
    Fetch a dataset and return a DataLoader with an ImbalancedDatasetSampler used to
    sample an imbalanced dataset.
    """
   logger.info("Get data loader")

   data = pd.read_csv(os.path.join(training_dir, filename))
   labels = data.LABEL.values
   features = data.loc[:, data.columns != "LABEL"].values

   tensor_labels = torch.tensor(labels)
   tensor_features = torch.tensor(features)

   dataset = TensorDataset(tensor_features, tensor_labels)

   tensor_dataloader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=ImbalancedDatasetSampler(dataset, labels))

   return tensor_dataloader
```
