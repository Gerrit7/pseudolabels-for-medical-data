import torch

from medmnist.info import INFO
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from Classes.Prepare_MedMNIST import PrepareMedMNIST

def main(input_args):
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
    '''
    # ************************************** train on gpu if possible **********************************************************************************
    
    if torch.cuda.is_available():     
        print('used GPU: ' + torch.cuda.get_device_name(0))
        device = torch.device("cuda:0")
        kwar = {'num_workers': 8, 'pin_memory': True}
        cpu = torch.device("cpu")
    
    else:
        print("Warning: CUDA not found, CPU only.")
        device = torch.device("cpu")
        kwar = {}
        cpu = torch.device("cpu")

    # ************************************** Create Dataloader  *****************************************************************************************
    
    dataset_info = INFO[input_args["data_name"]]

    prepare_medmnist = PrepareMedMNIST(input_args, dataset_info)

    dataloader_train_labeled = prepare_medmnist.getSubLoader('train', labeled=True)
    dataloader_train_unlabeled = prepare_medmnist.getSubLoader('train', labeled=False)
    dataloader_val = prepare_medmnist.getDataLoader('val')
    dataloader_test = prepare_medmnist.getDataLoader('test')

    # ************************************** Training  ***************************************************************************************************
    if input_args["task_input"] == "BaseLine":
        print("==> Baseline-Training...")

    elif input_args["task_input"] == "NoisyStudent":
        print("==> NoisyStudent-Training...")

    elif input_args["task_input"] == "MTSS":
        print("==> MTSS-Training...")

    elif input_args["task_input"] == "Pseudolabel":
        print("==> Pseudolabel-Training...")
    
    


