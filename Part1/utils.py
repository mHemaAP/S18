import matplotlib.pyplot as plt
#%matplotlib inline
import torchvision.transforms as T
import torch
import torch.nn as nn
import random

def show_sample_images(train_dataset):

    idx = random.randint(0, 500)
    fig, arr = plt.subplots(1, 2, figsize=(10, 10))
    arr[0].imshow(train_dataset[idx][0])
    arr[0].set_title('Image '+ str(1))
    arr[1].imshow(train_dataset[idx][1])
    arr[1].set_title('Masked Image '+ str(1))


# define plot losses
def plot_losses(train_losses, test_losses, title):
  #%matplotlib inline
  fig, axs = plt.subplots(1,2,figsize=(9,3))
  axs[0].plot(train_losses)
  axs[0].set_title("Training Loss", fontsize=12).set_position([.5, 0.8])
  axs[1].plot(test_losses)
  axs[1].set_title("Test Loss", fontsize=12).set_position([.5, 0.8])
  fig.suptitle(title, fontsize=16, x=0.5, y=1.10)


def show_image_results(model, test_dataloader):
    model.eval()
    model.to('cpu')
    transform = T.ToPILImage()
    (test_inputs, test_targets) = next(iter(test_dataloader))
    fig, arr = plt.subplots(6, 3, figsize=(15, 15)) # batch size 16

    for index in range(6):
      img = test_inputs[index].unsqueeze(0)
      pred_y = model(img)
      pred_y = nn.Softmax(dim=1)(pred_y)
      pred_mask = pred_y.argmax(dim=1)
      pred_mask = pred_mask.unsqueeze(1).to(torch.float)


      arr[index,0].imshow(transform(test_inputs[index]))
      arr[index,0].set_title('Processed Image')
      arr[index,1].imshow(transform(test_targets[index].float()))
      arr[index,1].set_title('Actual Masked Image ')
      arr[index,2].imshow(pred_mask.squeeze(0).squeeze(0))
      arr[index,2].set_title('Predicted Masked Image ')