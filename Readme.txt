Project Title:

Pytorch model using transfer learning from vgg16.

About:

The model has been used for distinguishing between Bee and Ants using pretrained vgg16 model, the alternation done here
is converting the last layer yielding 2 outputs. the model has been loaded using models module from torchvision library.
I have freezed the convolution layer by using the below command.

for param in model.features.parameters():
  param.requires_grad = False

Only the classifier section of the model has been changed as alexnet gives 1000 ouputs but we only require 2 as for 
distinguishing between ants and bees, so i have performed the below command for alteration.

n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs,len(classes))
model.classifier[6] = last_layer

The model gives around 91% of validation accuracy at about 4th epoch and starts to fall down.


Dataset used:
The Dataset used is the random created dataset available on https://github.com/jaddoescad/ants_and_bees.git
The dataset has 244 training images and 153 validation images of ants and bees. the data is of dimension 3 x 24 x 24 . 
AS the dataset is small, the model is trained using only 10 epochs. The model gives the training accuracy of 89% and
validation accuracy of 90%. The augmentation done here are :

transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
transforms.RandomAffine(0, shear = 10 , scale=(0.8,1.2))
transforms.ColorJitter(brightness = 1, contrast = 0.2, saturation = 0.2)
transforms.ToTensor()
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

Library used:

Torch
Numpy
matplotlib.pyplot
torch.nn
sklearn
torchvision
PIL.ImageOps