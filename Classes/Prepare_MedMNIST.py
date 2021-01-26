import torch
import torch.utils.data as data
import torchvision.transforms as transforms  
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal

import random
import numpy as np

class PrepareMedMNIST:
    def __init__(self, input_args, dataset_info):

        '''
        input args
            "data_name"         --> name of dataset                                                                                     --> string
            "data_root"         --> folder of medmnist data                                                                             --> string
            "output_root"       --> folder for saving results                                                                           --> string
            "n_epochs"          --> n of epochs in training                                                                             --> int
            "batch_size"        --> batch size                                                                                          --> int
            "learning_rate"     --> learning rate of the optimizer                                                                      --> float
            "momentum"          --> momentum of optimizer                                                                               --> float
            "train_size"        --> percentage of data for training                                                                     --> int
            "weight_decay"      --> weight decay                                                                                        --> float
            "model"             --> used model architecture                                                                             --> string
            "n_studentnets"     --> n of studentnet iterations in pseudolabeling                                                        --> int
            "operation"         --> train a model (False) or make predictions (True)                                                    --> boolean
            "task"              --> task: "Pseudolabel", "MTSS", "NoisyStudent", "Baseline"                                             --> string
            "optimizer"         --> optimizer                                                                                           --> string
            "LR_decay"          --> decay of learning rate: default 0                                                                   --> float
            "LR_milestones"     --> milestones for lr decay                                                                             --> int
            "Loss_function"     --> loss function                                                                                       --> string
            "augmentations"     --> list of augmenations                                                                                --> list
            "download"          --> download the data                                                                                   --> boolean
        
        dataset_info
            "description"       --> Description of dataset
            "url"               --> download url
            "MD5"               -->
            "task"              --> dataset task: "multi-class", "binary-class", "ordinal regression", "multi-label, binary-class"
            "label":            --> labels dictionary with class number and description
            "n_channels"        --> binary image (1), rgb image (3)
            "n_samples":        --> number of dataset splits: "train": XX, "val": XX, "test": XX
            "license":          --> licence for usage
        '''
        self.input_args = input_args
        self.dataset_info = dataset_info

        # create transformations of dataset 
        transform = self.createTransform(augmentations=self.input_args["augmentations"])

        # define dataset for training/ validation/ testing
        self.dataset_train   = self.prepareDataSet('train', transform)
        self.dataset_val     = self.prepareDataSet('train', transform)
        self.dataset_test     = self.prepareDataSet('train', transform)

        # split dataset if needed for training/ validation/ testing
        self.dataset_train_labeled, self.dataset_train_unlabeled  = self.splitDataset(self.dataset_train)

        # create dataloader for training/ validation/ testing
        self.dataloader_train = self.createDataLoader(self.dataset_train)
        self.dataloader_val = self.createDataLoader(self.dataset_val)
        self.dataloader_test = self.createDataLoader(self.dataset_test)

        self.dataloader_train_labeled = self.createDataLoader(self.dataset_train_labeled)
        if len(self.dataset_train_unlabeled) > 0:
            self.dataloader_train_unlabeled = self.createDataLoader(self.dataset_train_unlabeled)
        


    def createTransform(self, image_size=32, augmentations=[]):
        aug_values = {
            "CenterCrop"   : {"size": 10},
            "ColorJitter"  : {"brightness": 0, "contrast": 0, "saturation": 0, "hue": 0},
            "GaussianBlur" : {"kernel": [3,3], "sigma" : 0.1},
            "Normalize"    : {"mean": [0.5], "std": [0.5]},
            "RandomHorizontalFlip" : {"probability": 0.5},
            "RandomVerticalFlip" : {"probability": 0.5},
            "RandomRotation" : {"degrees": [-20, 20]}	
        }

        tranform_compose_list = [transforms.ToTensor()]
        if self.input_args["model"] == "EfficientNet-b0" or self.input_args["model"] == "EfficientNet-b1" or self.input_args["model"] == "EfficientNet-b7":
            tranform_compose_list.append(transforms.Resize(256))
        
        for aug in self.input_args["augmentations"]:
            if aug == "centerCrop":
                tranform_compose_list.append(transforms.CenterCrop(
                            aug_values["CenterCrop"]["size"]))
            elif aug == "colorJitter":
                tranform_compose_list.append(transforms.ColorJitter(
                            brightness=aug_values["ColorJitter"]["brightness"], 
                            contrast=aug_values["ColorJitter"]["contrast"],
                            saturation=aug_values["ColorJitter"]["saturation"], 
                            hue=aug_values["ColorJitter"]["hue"]))
            elif aug == "gaussianBlur":
                tranform_compose_list.append(transforms.GaussianBlur(
                            kernel_size=aug_values["GaussianBlur"]["kernel"], 
                            sigma=aug_values["GaussianBlur"]["sigma"]))
            elif aug =="normalize":
                tranform_compose_list.append(transforms.Normalize(
                            mean=aug_values["Normalize"]["mean"], 
                            std=aug_values["Normalize"]["std"]))
            elif aug =="randomHorizontalFlip":
                tranform_compose_list.append(transforms.RandomHorizontalFlip(
                            p=aug_values["RandomHorizontalFlip"]["probability"]))
            elif aug =="randomVerticalFlip":
                tranform_compose_list.append(transforms.RandomVerticalFlip(
                            p=aug_values["RandomVerticalFlip"]["probability"]))
            elif aug =="randomRotation":
                tranform_compose_list.append(transforms.RandomRotation(
                            degrees=aug_values["RandomRotation"]["degrees"]))
            #else:
                #print("augmentation not found!")
        
        transform = transforms.Compose(tranform_compose_list)
        return transform

    def prepareDataSet(self, split, transform):

        flag_to_class = {
            "pathmnist": PathMNIST,
            "chestmnist": ChestMNIST,
            "dermamnist": DermaMNIST,
            "octmnist": OCTMNIST,
            "pneumoniamnist": PneumoniaMNIST,
            "retinamnist": RetinaMNIST,
            "breastmnist": BreastMNIST,
            "organmnist_axial": OrganMNISTAxial,
            "organmnist_coronal": OrganMNISTCoronal,
            "organmnist_sagittal": OrganMNISTSagittal,
        }
        DataClass = flag_to_class[self.input_args["data_name"]]

        dataset = DataClass(root=self.input_args["data_root"],
                            split=split,
                            transform=transform,
                            download=self.input_args["download"])

        return dataset

    def createDataLoader(self, dataset):
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=self.input_args["batch_size"],
                                    shuffle=True)
        
        return data_loader


    def splitDataset(self, dataset):
        images, labels = dataset.get_data()

        vars()["train_counter"] = np.zeros(len(self.dataset_info['label']))
        vars()["train_class_list"] = []
        for idx in range(len(self.dataset_info['label'])):
            vars()["train_class_list"].append([])
                
        for i in range(len(labels)):
            vars()["train_counter"][int(labels[i])] +=1
            vars()["train_class_list"][int(labels[i])].append((images[i], labels[i]))

        dataset_labeled = []
        dataset_unlabeled = []
        for class_list in range(len(self.dataset_info['label'])):
            random_samples = random.sample(vars()["train_class_list"][class_list], int((self.dataset_info["n_samples"]["train"] * self.input_args["train_size"])/len(self.dataset_info['label'])))
            dataset_labeled = dataset_labeled + random_samples
        
        return dataset_labeled, dataset_unlabeled


    def getDataset(self, split):
        if split == "train":
            return self.dataset_train
        elif split == "val":
            return self.dataset_val
        elif split == "test":
            return self.dataset_test
        else:
            print("split param has to be: train, val or test!")
    
    def getDataLoader(self, split):
        if split == "train":
            return self.dataloader_train
        elif split == "val":
            return self.dataloader_val
        elif split == "test":
            return self.dataloader_test
        else:
            print("split param has to be: train, val or test!")

    def getSubLoader(self,split, labeled=True):
        if split == "train":
            if labeled == True:
                return self.dataloader_train_labeled
            else:
                return self.dataloader_train_unlabeled
        elif split == "val":
            if labeled == True:
                return self.dataloader_val_labeled
            else:
                return self.dataloader_val_unlabeled
        elif split == "test":
            if labeled == True:
                return self.dataloader_test_labeled
            else:
                return self.dataloader_test_unlabeled
        else:
            print("split param has to be: train, val or test!")

    def getSplitData(self, split):
        if split == "train":
            return [self.train_dataset_labeled, self.train_dataset_unlabeled]
        elif split == "val":
            return [self.val_dataset_labeled, self.val_dataset_unlabeled]
        elif split == "test":
            return [self.test_dataset_labeled, self.test_dataset_unlabeled]
        else:
            print("split param has to be: train, val or test!")
