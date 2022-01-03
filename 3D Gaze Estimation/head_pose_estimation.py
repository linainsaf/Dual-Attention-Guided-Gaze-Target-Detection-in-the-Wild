import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def createModel_resnet34(out_1, out_2):
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False

    # Creating 3 Linear connected layers which can be trained
    fc1 = nn.Linear(num_ftrs, out_1)
    fc2 = nn.Linear(out_1, out_2)
    fc3 = nn.Linear(out_2, 2)

    layers = [fc1, fc2, fc3]
    for linearLayer in layers:
        # Applying He initialization to all layers
        nn.init.kaiming_uniform_(linearLayer.weight, nonlinearity='leaky_relu')

    # Setting Resnet's fully connected layer to our collection of three Linear layers with nn.Sequential
    model.fc = nn.Sequential(fc1, nn.LeakyReLU(), fc2, nn.LeakyReLU(), fc3)
    model.double()  # double to set variables to double
    # Sending the device to the GPU if avaliable
    model.to(device)

    return model


class headposeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgs[idx]
        image = Image.fromarray(image)
        '''
        landmarks = np.zeros(2)
        landmarks=np.zeros(2)
        landmarks[0]= ex[0]
        landmarks[1]= ex[1]
        '''
        if self.transform:
            image = self.transform(image)

        sample = {'image': image}

        return sample


def head_pose_extraction(head_imgs):
    batch_size = 128  # 32
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.double()),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    head_tensor = headposeDataset(imgs=head_imgs, transform=preprocess)
    head_tensor_batched = torch.utils.data.DataLoader(head_tensor, batch_size=batch_size, num_workers=2)

    # head pose prediction
    model_hp = createModel_resnet34(512, 128)
    model_hp.load_state_dict(
        torch.load("drive/MyDrive/MLA/head_pose_extractor.pth"))  # ,map_location=torch.device('cpu')))
    model_hp.eval()

    prediction_hp = torch.zeros((1, 2), dtype=torch.int32, device='cuda')

    for i, batch in enumerate(head_tensor_batched):
        prediction = model_hp(batch["image"].to(device))
        prediction_hp = torch.cat((prediction_hp, prediction), 0)
    prediction_hp = prediction_hp[1:, :]
    return prediction_hp