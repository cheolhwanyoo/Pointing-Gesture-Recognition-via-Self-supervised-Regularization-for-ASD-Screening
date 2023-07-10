import torchvision.transforms as transforms
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from simsiam import loader

def get_dataloaders(args):

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    # Handle data
    transformer_tra = transforms.Compose([
                            transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            # additional augmentation
                            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                            # transforms.RandomGrayscale(p=0.2),
                            # transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                            ])


    transformer_eval = transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
                            ])

    dataset_training = PointDataset(transform=transformer_tra, stage='train', args=args)
    dataset_val = PointDataset(transform=transformer_eval, stage='val', args=args)

    datasets = {'train': dataset_training, 'val': dataset_val}

    dataloaders = {
                    'train': DataLoader(datasets['train'], batch_size=args.batchsize, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True, pin_memory=True),
                    'val': DataLoader(datasets['val'], batch_size=1, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False, pin_memory=True),
                  }

    return dataloaders


class PointDataset(Dataset):

    def __init__(self, root_dir='dir/nturgbd_rgb/pointing_binaryDB_hand/',
                 transform=None,
                 stage='train',
                 args=None):

        self.args = args
        self.transform = transform
        self.rgb_list = []
        self.labels = []

        # positive list
        self.input_list_pos = []
        self.labels_pos = []

        # negative list
        self.input_list_neg = []
        self.labels_neg = []

        ## NTU DB ##
        if stage == 'train' or stage == 'val':
            basedir_pos = os.path.join(root_dir, '0_refined_{}'.format(stage))
            basedir_neg = os.path.join(root_dir, '1_refined_{}'.format(stage))

            self.input_list_pos += [os.path.join(basedir_pos, f) for f in sorted(os.listdir(basedir_pos)) if
                                        f.split(".")[-1] == "png" or f.split(".")[-1] == "jpg"]

            self.input_list_neg += [os.path.join(basedir_neg, f) for f in sorted(os.listdir(basedir_neg)) if
                                         f.split(".")[-1] == "png" or f.split(".")[-1] == "jpg"]

        for i in range(len(self.input_list_pos)):
            self.labels_pos.append(0)

        self.labels_neg = []
        for i in range(len(self.input_list_neg)):
            self.labels_neg.append(1)

        # Total list
        self.rgb_list = self.input_list_pos + self.input_list_neg
        self.labels = self.labels_pos + self.labels_neg

        if stage =='train':
            self.rgb_list, self.labels = shuffle(self.rgb_list, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        rgbpath = self.rgb_list[idx]

        img = Image.open(rgbpath)
        label = self.labels[idx]

        if self.transform:
            if self.args.SSL == "None":
                img = self.transform(img)
                sample = {'rgb': img, 'label': label, 'rgbpath': rgbpath}
            else:
                img_copy = img.copy()
                img_aux = self.transform(img_copy)
                img = self.transform(img)
                sample = {'rgb': img, 'label': label, 'rgbpath': rgbpath, 'rgb_aux': img_aux}

        return sample

    def img_noramlize(self, np_clip):

        # Div by 255
        np_clip /= 255.

        # Normalization
        np_clip -= np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # mean
        np_clip /= np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 3)  # std

        return np_clip

