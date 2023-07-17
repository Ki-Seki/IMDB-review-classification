import torch


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    running_loss, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        correct +=  ((pred>0.5).long() == y.long()).sum().item()

    return running_loss/num_batches, correct/num_samples

def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    running_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            running_loss += loss_fn(pred, y).item()
            correct += ((pred>0.5).long() == y.long()).sum().item()
    
    return running_loss/num_batches, correct/num_samples
