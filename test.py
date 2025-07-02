import torch

import dataloader
    
def accuracy(y_pred, y):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
    cmp = y_pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(model, data_iter, device):
    if isinstance(model, torch.nn.Module):
        model.eval()
    acc_preds, total_preds = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            acc_preds += accuracy(model(X), y)
            total_preds += y.numel()
    return acc_preds / total_preds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = torch.load("cifar.pth")
    params.eval()
    
    batch_size = 256
    num_workers = 4
    
    _, test_loader = dataloader.load_data(batch_size, num_workers)

    
    acc = evaluate_accuracy(params, test_loader, device)
    print(f"Test accuracy: {acc:.3f}")