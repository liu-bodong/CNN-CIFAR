import torch
import torch.nn as nn
import torch.optim as optim

import model
import dataloader
import test

def train(model, train_loader, epochs, device, save=False, debug=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters())
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        num_samples = 0
        
        for (x, y) in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            loss.backward()
            optimizer.step()
            
            if debug:
                with torch.no_grad():
                    loss = loss.float()
                    train_loss += loss
                    train_acc += test.accuracy(y_pred, y)
                    num_samples += y.numel()
                    #
                
        if debug and epoch % 1 == 0:
            train_loss /= num_samples
            train_acc /= num_samples
            print(f'Epoch {epoch}, '
                  f'train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}')
        
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device:", device)
    
    batch_size = 256
    num_workers = 4
    train_loader, test_loader = dataloader.load_data(1, num_workers)
    model = model.CNN()
    
    # print(model)
    
    # print("net has {} parametres in total".format(
    #     sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # for x, y in train_loader:
    #     model(x)
    #     break
    
    
    
    parameters = train(model=model, train_loader=train_loader, epochs=10, device=device, debug=True)
    
    torch.save(model.state_dict(), 'cifar.pth')
    
    test_acc = test.evaluate_accuracy(model, test_loader, device)
    
    print(f'Test accuracy {test_acc:.5f}')
    