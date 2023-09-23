import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# define losses
ce_loss_fn = nn.CrossEntropyLoss()

def dice_loss_fn(pred,target,n_classes=3):
  smooth = 0.001
  pred = F.softmax(pred,dim=1).float().flatten(0,1) # (96,128,128)-> 3 * 32
  target = F.one_hot(target, n_classes).squeeze(1).permute(0, 3, 1, 2).float().flatten(0,1) # (96,128,128) -> 3 * 32
  assert pred.size() == pred.size(), "sizes do not match"

  intersection = 2 * (pred * target).sum(dim=(-1, -2)) # 96
  union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) #96
  #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

  dice = (intersection + smooth) / ( union + smooth)

  return 1 - dice.mean()

def get_loss_function(loss_type):
    if(loss_type == 'CE'):
        loss_function = ce_loss_fn
    elif(loss_type == 'Dice_Loss'):
        loss_function = dice_loss_fn

    return loss_function

def train(model, loss_fn, device, train_dataloader, optimizer):

  model.train()
  train_loss = 0

  for batch, (X,y) in enumerate(train_dataloader):
    X,y = X.to(device), y.to(device)
    y_pred = model(X)

    loss = loss_fn(y_pred,y.squeeze(1))
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss = train_loss / len(train_dataloader)

  return train_loss

def test(model, loss_fn, device, test_dataloader):

  model.eval()
  test_loss = 0

  with torch.inference_mode():
    for batch, (X,y) in enumerate(test_dataloader):
      X,y = X.to(device), y.to(device)
      y_pred = model(X)

      loss = loss_fn(y_pred,y.squeeze(1))
      test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)

  return test_loss


def run_training(model, loss_type, num_epochs, train_dataloader, test_dataloader, device):
    optimizer = torch.optim.Adam(params = model.parameters(), lr=0.001)
    loss_function = get_loss_function(loss_type)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, loss_function, device, train_dataloader, optimizer)
        test_loss = test(model, loss_function, device, test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(
                f"Epoch: {epoch+1}   | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} "
                )

    return train_losses, test_losses