import torch
import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, lr, mode=None):
    ## Construct optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    ## Set phase
    model.train()
    
    ## Train start
    total_loss = 0.
    for data, target in train_loader:
        ## data.shape = (batch_size, 22)
        ## target.shape = (batch_size, 1)
        ## initilize gradient
        optimizer.zero_grad()
        if mode=='mha_dnn':
            z = torch.zeros((target.shape[0], 1)).cuda()
            z_target = torch.cat([z, target], 1) # [<s>, rate, waf, sa, time]
            target_z = torch.cat([target, z], 1) # [rate, waf, sa, time, <e>]
            output, _ = model(data, z_target)
            loss = F.mse_loss(output, target_z)
        else:
            ## predict
            output, _ = model(data, target) # output.shape = (batch_size, 1)
            ## loss
            loss = F.mse_loss(output, target)
        ## backpropagation
        loss.backward()
        optimizer.step()
        ## Logging
        total_loss += loss.item()
    total_loss /= len(train_loader)
    return total_loss

def valid(model, valid_loader, mode=None):
    ## Set phase
    model.eval()
    
    ## Valid start    
    total_loss = 0.
    outputs = torch.Tensor().cuda()
    with torch.no_grad():
        for data, target in valid_loader:
            if mode == 'mha_dnn':
                output, _ = model(data)
                # z = torch.zeros((target.shape[0], 1)).cuda()
                # target = torch.cat([target, z], 1) # [rate, waf, sa, time, <e>]
            else:
                output, _ = model(data, target)
            loss = F.mse_loss(output, target) # mean squared error
            total_loss += loss.item()
            outputs = torch.cat((outputs, output))
    total_loss /= len(valid_loader)
    return total_loss, outputs