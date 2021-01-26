'''
This function is the Selector GUI for CNN Training, the selected entries start the main function
'''
from main import main

from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
from tkinter import *


def switch(value):
    global switch
  
def sel():
   selection = "You selected the net-structure " + str(var_model.get())
   #label.config(text = selection)
    
def sel_task():
    selection_task = "You selected the task " + str(var_task.get())
    if var_task.get() != "Pseudolabel":
         e_count_stud.config(state="disabled")
    else:
        e_count_stud.config(state="normal")
    #     var_batchsize.set(1)
    #     e2.config(state="disabled") 
    #     e3.config(state="disabled")
    #     e4.config(state="disabled")
    #     e6.config(state="disabled")

    # else:
    #     e1.config(state='normal')
    #     e2.config(state='normal')
    #     var_batchsize.set(8)
    #     e3.config(state='normal')
    #     e4.config(state='normal')
    #     e6.config(state='normal')
    #label_model.config(text = selection_task)

def sel_optimizer():
    selection_optimizer = "You selected the optimizer " + str(var_optimizer.get())
    #label_optimizer.config(text = selection_optimizer)
    
def sel_lossfun():
    selection_lossfun = "You selected the loss function " + str(var_lossfun.get())
    #label_lossfun.config(text = selection_lossfun)
    
def sel_medmnist():
    D11_var.set(False)
    variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var]
    datasets = ["Pathmnist", "Chestmnist", "Dermamnist", "OCTmnist", "Pneumoniamnist", "Retinamnist", "Breastmnist", "Organmnist_axial", "Organmnist_coronal", "Organmnist_sagittal"]
    selection_dataset = "You selected the dataset/s " + ', '.join([str(datasets[x]) for x in range(len(datasets)) if variables[x].get()!=""])
    label_dataset.config(text = selection_dataset)
    global data_name 
    data_name = []
    for var in variables:
        print(var.get())
        if var.get()!= "":
            data_name.append(var.get())
 
def sel_cifar10():
    variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var]
    for i in range(len(variables)):
        variables[i].set(False)
    selection_dataset = "You selected the dataset Cifar10"
    label_dataset.config(text = selection_dataset)
      
def submit(): 
    # some tasks have to be selected before the app can continue
    if data_name == []:
        print("Please select at least one dataset!")
    elif var_model.get() == "":
        print("Please select Net Structure!")
    elif var_task.get() == "":
        print("Please select Task!")
    elif var_optimizer.get() == "":
        print("Please select an optimizer!")
    elif var_lossfun.get() == "":
        print("Please select a loss function")
    else:
   
        root.destroy()
        # start the main function for every selected dataset
        
        for i in range(len(data_name)):
            input_args = {
                "data_name"         : data_name[i],
                "data_root"         : data_root_complete.get(),
                "output_root"       : output_root_complete.get(),
                "n_epochs"          : var_epochs.get(),
                "batch_size"        : var_batchsize.get(),
                "learning_rate"     : var_lr.get(),
                "momentum"          : var_momentum.get(),
                "train_size"        : var_percent.get()*0.01,
                "weight_decay"      : var_weightdecay.get(),
                "model"             : var_model.get(),
                "n_studentnets"     : var_count_stud.get(),
                "operation"         : var_switch.get(),
                "task"              : var_task.get(),
                "optimizer"         : var_optimizer.get(),
                "LR_decay"          : var_decayLr.get(),
                "LR_milestones"     : var_milestone.get(),
                "Loss_function"     : var_lossfun.get(),
                "augmentations"     : [c1_var.get(), c2_var.get(), c3_var.get(), c4_var.get(), c5_var.get(), c6_var.get(), c7_var.get()],
                "download"          : True
            }
            
            
            main(input_args)
        

def browse_data():
    global data_root
    filename = filedialog.askdirectory()
    data_root.set(filename.split("/")[-2] + "/" + filename.split("/")[-1])  
    data_root_complete.set(filename)
        
def browse_output():
    global output_root
    filename = filedialog.askdirectory()
    output_root.set(filename.split("/")[-2] + "/" + filename.split("/")[-1])
    output_root_complete.set(filename)

def printValues(choices):
    for name, var in choices.items():
        print("%s: %s" % (name, var.get()))
        
data_name = []

root = tk.Tk()
root.title('CNN Selector')
root.geometry("1200x600")

D1_var = StringVar()
D2_var = StringVar()
D3_var = StringVar()
D4_var = StringVar()
D5_var = StringVar()
D6_var = StringVar()
D7_var = StringVar()
D8_var = StringVar()
D9_var = StringVar()
D10_var = StringVar()
D11_var = StringVar()

# Area for selecting the datasets
menubutton = tk.Menubutton(text="Select Dataset",indicatoron=True, borderwidth=1, relief="raised")
menu = tk.Menu(menubutton, tearoff=False)
menubutton.configure(menu=menu)
menubutton.grid(row=0 ,column = 4, columnspan=4, padx='5', pady='5',sticky='ew')

labels = ["PathMnist", "ChestMnist", "Dermamnist", "OCTMnist", "PneumoniaMnist", "RetinaMnist", "BreastMnist", "OrganMnist axial", "OrganMnist coronal", "OrganMnist sagittal"]
onvalues = ["pathmnist", "chestmnist", "dermamnist","octmnist", "pneumoniamnist", "retinamnist", "breastmnist", "organmnist_axial", "organmnist_coronal", "organmnist_sagittal"]
variables = [D1_var, D2_var, D3_var, D4_var, D5_var, D6_var, D7_var, D8_var, D9_var, D10_var, D11_var]
for choice in range(len(labels)):
    menu.add_checkbutton(label=labels[choice], variable=variables[choice], 
                         onvalue=onvalues[choice], offvalue="", 
                         command=sel_medmnist)


label_dataset = Label(root, text="Please select dataset!",height = 4, wraplength=250)
label_dataset.grid(row=0, padx='5', pady='5',sticky='ew')

# area for browsing for data and output directories
Label(root, text = "Directories", bd=1, relief="solid", width = 56, height = 2).grid(row=1 ,column = 0, padx='0', pady='0',sticky='ew',columnspan=3)
data_root = StringVar()
data_root_complete = StringVar()
output_root = StringVar()
output_root_complete = StringVar()
data_root_label = Label(master=root,textvariable=data_root, wraplength=250, justify=LEFT).grid(row=2 ,column = 1, padx='5', pady='5',sticky='ew')
output_root_label = Label(master=root,textvariable=output_root, wraplength=250, justify=LEFT).grid(row=3 ,column = 1, padx='5', pady='5',sticky='ew')
browse_data_button = Button(text="Browse Dataset Root", command=browse_data).grid(row=2 ,column = 0, padx='5', pady='5',sticky='ew')
browse_output_button = Button(text="Browse Output Root", command=browse_output).grid(row=3 ,column = 0, padx='5', pady='5',sticky='ew')

# area for selecting the operation type (training - prediction)
Label(root, text = "Operation", bd=1, relief="solid", width = 56, height = 2).grid(row=4 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')
var_switch = BooleanVar()
var_switch.set(False)
tk.Label(root, text = "training", bd=1, relief="solid", width = 15, height = 2).grid(row=5 ,column = 0, padx='5', pady='5',sticky='ew')
switch = tk.Scale(orient = tk.HORIZONTAL,length = 50,to = 1, variable = var_switch, showvalue = False,sliderlength = 25,command = switch)
switch.grid(row=5 ,column = 1, padx='5', pady='5',sticky='ew')
tk.Label(root, text = "prediction", bd=1, relief="solid", width = 15, height = 2).grid(row=5 ,column = 2, padx='5', pady='5',sticky='ew')



# area for selecting the used net structure
Label(root, text = "NetStructure", bd=1, relief="solid", width = 56, height = 2).grid(row=7 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var_model = StringVar()
R1 = Radiobutton(root, text="ResNet-18", variable=var_model, value="Resnet18",
                  command=sel)
R1.grid(row=8 ,column = 0, padx='5', pady='5',sticky='ew')

R2 = Radiobutton(root, text="ResNet-50", variable=var_model, value="ResNet50",
                  command=sel)
R2.grid(row=9 ,column = 0, padx='5', pady='5',sticky='ew')

R3 = Radiobutton(root, text="EfficientNet-b0", variable=var_model, value="EfficientNet-b0",
                  command=sel)
R3.grid(row=8 ,column = 1, padx='5', pady='5',sticky='ew')

R4 = Radiobutton(root, text="EfficientNet-b1", variable=var_model, value="EfficientNet-b1",
                  command=sel)
R4.grid(row=9 ,column = 1, padx='5', pady='5',sticky='ew')

R5 = Radiobutton(root, text="EfficientNet-b7", variable=var_model, value="EfficientNet-b7",
                  command=sel)
R5.grid(row=10 ,column = 1, padx='5', pady='5',sticky='ew')

R7 = Radiobutton(root, text="AlexNet", variable=var_model, value="AlexNet",
                  command=sel)
R7.grid(row=8 ,column = 2, padx='5', pady='5',sticky='ew')


# area for selecting the used optimizer
Label(root, text = "Optimizer", bd=1, relief="solid", width = 56, height = 2).grid(row=11 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var_optimizer = StringVar()
Op1 = Radiobutton(root, text="SGD", variable=var_optimizer, value="SGD",
                  command=sel_optimizer)
Op1.grid(row=12 ,column = 0, padx='5', pady='5',sticky='ew')

Op2 = Radiobutton(root, text="Adam", variable=var_optimizer, value="Adam",
                  command=sel_optimizer)
Op2.grid(row=12 ,column = 1, padx='5', pady='5',sticky='ew')

Op3 = Radiobutton(root, text="RMSProp", variable=var_optimizer, value="RMSprop",
                  command=sel_optimizer)
Op3.grid(row=12 ,column = 2, padx='5', pady='5',sticky='ew')


# area for selecting the used loss function
Label(root, text = "Loss Function", bd=1, relief="solid", width = 56, height = 2).grid(row=13 ,column = 0, columnspan=3, padx='0', pady='0',sticky='ew')

var_lossfun = StringVar()
Loss1 = Radiobutton(root, text="CrossEntropyLoss", variable=var_lossfun, value="crossentropyloss",
                  command=sel_lossfun)
Loss1.grid(row=14 ,column = 0, padx='5', pady='5',sticky='ew')

Loss2 = Radiobutton(root, text="BCE with Logitsloss", variable=var_lossfun, value="bce",
                  command=sel_lossfun)
Loss2.grid(row=15 ,column = 0, padx='5', pady='5',sticky='ew')

Loss3 = Radiobutton(root, text="Max likelihood estimation", variable=var_lossfun, value="MLE",
                  command=sel_lossfun)
Loss3.grid(row=14 ,column = 1, padx='5', pady='5',sticky='ew')

Loss3 = Radiobutton(root, text="Mean Squared Error", variable=var_lossfun, value="MSE",
                  command=sel_lossfun)
Loss3.grid(row=15 ,column = 1, padx='5', pady='5',sticky='ew')



# area for selecting the used augmentations (multiple selection possible)
Label(root, text = "Augmentations", bd=1, relief="solid", width = 56, height = 2).grid(row=1 ,column = 4,columnspan=4, padx='0', pady='0',sticky='ew')
c1_var = StringVar()
c2_var = StringVar()
c3_var = StringVar()
c4_var = StringVar()
c5_var = StringVar()
c6_var = StringVar()
c7_var = StringVar()

c1 = Checkbutton(root, text='centerCrop',variable=c1_var, onvalue="centerCrop", offvalue="")
c1.grid(row=2 ,column = 4, padx='5', pady='5',sticky='ew')

c2 = Checkbutton(root, text='colorJitter',variable=c2_var, onvalue="colorJitter", offvalue="")
c2.grid(row=3 ,column = 4, padx='5', pady='5',sticky='ew')

c3 = Checkbutton(root, text='gaussianBlur',variable=c3_var, onvalue="gaussianBlur", offvalue="")
c3.grid(row=4 ,column = 4, padx='5', pady='5',sticky='ew')

c4 = Checkbutton(root, text='normalize',variable=c4_var, onvalue="normalize", offvalue="")
c4.grid(row=2 ,column = 5, padx='5', pady='5',sticky='ew')

c5 = Checkbutton(root, text='randHorFlip',variable=c5_var, onvalue="randomHorizontalFlip", offvalue="")
c5.grid(row=3 ,column = 5, padx='5', pady='5',sticky='ew')

c6 = Checkbutton(root, text='randVertFlip',variable=c6_var, onvalue="randomVerticalFlip", offvalue="")
c6.grid(row=4 ,column = 5, padx='5', pady='5',sticky='ew')

c7 = Checkbutton(root, text='randRot',variable=c7_var, onvalue="randomRotation", offvalue="")
c7.grid(row=2 ,column = 6, padx='5', pady='5',sticky='ew')



# area for selecting the used tasks
Label(root, text = "Task", bd=1, relief="solid", width = 56, height = 2).grid(row=4 ,column =4, columnspan=4, padx='0', pady='0',sticky='ew')

var_task = StringVar()
M1 = Radiobutton(root, text="Pseudolabel", variable=var_task, value="Pseudolabel",
                  command=sel_task)
M1.grid(row=5 ,column = 4, padx='5', pady='5',sticky='ew')

M2 = Radiobutton(root, text="MTSS", variable=var_task, value="MTSS",
                  command=sel_task)
M2.grid(row=6 ,column = 4, padx='5', pady='5',sticky='ew')

M3 = Radiobutton(root, text="NoisyStudent", variable=var_task, value="NoisyStudent",
                  command=sel_task)
M3.grid(row=5 ,column = 5, padx='5', pady='5',sticky='ew')

M4 = Radiobutton(root, text="BaseLine", variable=var_task, value="BaseLine",
                  command=sel_task)
M4.grid(row=6 ,column = 5, padx='5', pady='5',sticky='ew')



# area for selecting the used parameters
Label(root, text = "Parameters", bd=1, relief="solid", width = 56, height = 2).grid(row=7 ,column = 4, columnspan=4, padx='0', pady='0',sticky='ew')

Label(root, text="Epochs").grid(row=8 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Batch Size").grid(row=9 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Learning Rate").grid(row=10 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Decay Learning Rate").grid(row=11 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Milestone Count").grid(row=12 ,column = 4, padx='5', pady='5',sticky='ew')
Label(root, text="Momentum").grid(row=8 ,column = 6, padx='5', pady='5',sticky='ew')
Label(root, text="% of trainset").grid(row=9 ,column = 6, padx='5', pady='5',sticky='ew')
Label(root, text="Weight Decay").grid(row=10 ,column = 6, padx='5', pady='5',sticky='ew')
Label(root, text="Count of Student Nets").grid(row=11 ,column = 6, padx='5', pady='5',sticky='ew')

var_epochs = IntVar()
var_batchsize = IntVar()
var_lr = DoubleVar() 
var_decayLr = DoubleVar()
var_milestone = IntVar()
var_momentum = DoubleVar() 
var_percent = IntVar()
var_weightdecay = DoubleVar() 
var_count_stud = IntVar()

e1 = Entry(root,textvariable=var_epochs)
e1.grid(row=8 ,column = 5, padx='5', pady='5',sticky='ew')
e2 = Entry(root,textvariable=var_batchsize)
e2.grid(row=9 ,column = 5, padx='5', pady='5',sticky='ew')
e3 = Entry(root,textvariable=var_lr)
e3.grid(row=10 ,column = 5, padx='5', pady='5',sticky='ew')
e7 = Entry(root,textvariable=var_decayLr)
e7.grid(row=11 ,column = 5, padx='5', pady='5',sticky='ew')
e8 = Entry(root,textvariable=var_milestone)
e8.grid(row=12 ,column = 5, padx='5', pady='5',sticky='ew')
e4 = Entry(root,textvariable=var_momentum)
e4.grid(row=8 ,column = 7, padx='5', pady='5',sticky='ew')
e5 = Entry(root,textvariable=var_percent)
e5.grid(row=9 ,column = 7, padx='5', pady='5',sticky='ew')
e6 = Entry(root,textvariable=var_weightdecay)
e6.grid(row=10 ,column = 7, padx='5', pady='5',sticky='ew')
e_count_stud = Entry(root,textvariable=var_count_stud)
e_count_stud.grid(row=11 ,column = 7, padx='5', pady='5',sticky='ew')



# submit button for starting the main function
SubmitButton = Button(root, text="Submit", command=submit)
SubmitButton.grid(row=14 ,column = 5, columnspan=2, padx='5', pady='5',sticky='ew')

#Setting Defaults
data_root.set("/DataSets/medmnist")
data_root_complete.set("/home/gerrit/Dokumente/Master_Thesis/DataSets/medmnist")
output_root.set("/Master-Thesis/TrainedNets/")
output_root_complete.set("/home/gerrit/Dokumente/Master_Thesis/TrainedNets/PseudoLabeling")
var_epochs.set(100)
var_batchsize.set(128)
var_lr.set(0.001)
var_decayLr.set(0.1)
var_milestone.set(0)
var_momentum.set(0.9)
var_percent.set(10)
var_weightdecay.set(0)
var_count_stud.set(3)

root.mainloop()

