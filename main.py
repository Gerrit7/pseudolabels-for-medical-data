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
    
    dataset_info = INFO[input_args["data_name"]]

    prepare_medmnist = PrepareMedMNIST(input_args, dataset_info)

    dataloader_train_labeled = prepare_medmnist.getSubLoader('train', labeled=True)

    print(len(dataloader_train_labeled.dataset))
    for i in range(len(dataloader_train_labeled.dataset)):
        print(dataloader_train_labeled.dataset[i][1])
    
    


