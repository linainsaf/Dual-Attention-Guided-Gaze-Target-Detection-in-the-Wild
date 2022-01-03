import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from head_pose_estimation import head_pose_extraction
from eye_features_extraction import get_eye_features
from gaze_prediction import mlp_gaze_estimation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def createModel_MLP(hidden, tensor_size):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(tensor_size, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 3),
    )

    model.double()  # double to set variables to double
    # Sending the device to the GPU if avaliable
    model.to(device)

    return model


class MLPDataset(Dataset):
    """prep_data_for_MLP."""

    def __init__(self, data_in, transform=None):
        self.data_in = data_in
        self.transform = transform

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data_in[idx]

        if self.transform:
              x = self.transform(x)
        return x


#we multiply this operator with the features extracted from the resnet
def l_op(l_imgs,r_imgs):
    l_imgs = torch.as_tensor(np.array(l_imgs)).cuda()
    r_imgs = torch.as_tensor(np.array(r_imgs)).cuda()
    eyes_b =l_imgs.sum(axis = (3,2,1)) +r_imgs.sum(axis = (3,2,1))
    lop = (eyes_b).bool()
    lop = lop.int()
    return lop

def lop_eyes_features( lop, eyes_features):
    x= eyes_features
    for i in range(len(lop)):
      x[i] = torch.mul(eyes_features[i],lop[i])
    return x


def mlp_gaze_estimation(prediction_hp, features_left, features_right, left_eye_imgs, right_eye_imgs):
  batch_size= 571
  #prepare head position input
  h = torch.as_tensor(prediction_hp).cuda()
  #prepare eyes features : concatenation + multiplication with l operator
  eyes_features = torch.cat((features_left, features_right),1)
  lop = l_op(left_eye_imgs, right_eye_imgs)
  # Multiply eyes features and L operator
  lop_EyesFeatures = lop_eyes_features(lop,eyes_features)
  # Concatenate eyes features with h
  input_data = torch.cat((h,eyes_features), 1)
  #Normalize the concatenated tensor
  input= input_data.clone().detach()
  input = ((input.T - input.mean(axis = 1))/input.std(axis = 1)).T

  #preprocess data
  preprocess = transforms.Compose([transforms.Lambda(lambda x: x.double())])
  dataset = MLPDataset(input, transform=preprocess)
  dataset_batched = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  #Model import
  tensor_size= len(features_left[0])*2+len(prediction_hp[0])
  model_gaze = createModel_MLP(256,tensor_size)
  model_gaze.load_state_dict(torch.load("models/MLP.pth"))#,map_location=torch.device('cpu')))
  model_gaze.eval()

  gaze_prediction =torch.zeros((1,3), dtype=torch.int32, device = 'cuda')

  for i, batch in enumerate(dataset_batched):
    prediction = model_gaze(batch.to(device))
    gaze_prediction = torch.cat((gaze_prediction,prediction), 0)
  gaze_prediction = gaze_prediction[1:,:]

  return gaze_prediction

def gaze_estimation(head_imgs, left_eye_imgs, right_eye_imgs):
    # head pose prediction
    prediction_hp =  head_pose_extraction(head_imgs)
    print("head pose estimated")

    # Get eye features
    features_left, features_right = get_eye_features(left_eye_imgs, right_eye_imgs)
    print("eyes features extracted")

    # gaze prediction
    gaze_prediction = mlp_gaze_estimation(prediction_hp, features_left, features_right, left_eye_imgs, right_eye_imgs)
    print("3D gaze predicted")

    return gaze_prediction