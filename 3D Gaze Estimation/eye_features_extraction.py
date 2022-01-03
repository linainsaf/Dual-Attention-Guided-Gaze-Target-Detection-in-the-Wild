import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class eyeDataset(Dataset):
    """eye landmark dataset."""

    def __init__(self, imgs, transform=None):

        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 0])
        image = self.imgs[idx]  # io.imread(img_name)
        image = Image.fromarray(image)
        sample = {'image': image}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def get_eye_features(left_eye_imgs, right_eye_imgs):
    # model import
    resnet = models.resnet18(pretrained=True)
    resnet18 = nn.Sequential(*(list(resnet.children())[:-1])) #take 8 layers
    resnet18.to(device)
    # transform data
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    # left eye
    l_eye_set = eyeDataset(imgs=left_eye_imgs,transform=preprocess)
    l_eye_loader = DataLoader(l_eye_set, batch_size=571, shuffle=False, num_workers=2)
    tensor_size = 512
    features_left = torch.zeros((1,tensor_size), dtype=torch.int32, device='cuda')

    for i in range(len(l_eye_loader)):
        l = next(iter(l_eye_loader))
        outputs_l = l['image'].to(device)
        left = resnet18(outputs_l).flatten(start_dim=1)
        features_left = torch.cat((features_left,left), 0)

    features_left = features_left[1:,:]

    # right eye
    r_eye_set = eyeDataset(imgs=right_eye_imgs, transform=preprocess)
    r_eye_loader = DataLoader(r_eye_set, batch_size=571, shuffle=False, num_workers=2)
    resnet18.eval()
    tensor_size = 512
    features_right = torch.zeros((1,tensor_size), dtype=torch.int32, device='cuda')
    for i in range(len(r_eye_loader)):
        r=next(iter(r_eye_loader))
        outputs_r=r['image'].to(device)
        right = resnet18(outputs_r).flatten(start_dim=1)# yields a tensor of size([batch_size, 2048])
        features_right = torch.cat((features_right, right), 0)

    features_right = features_right[1:, :]

    return features_left, features_right
