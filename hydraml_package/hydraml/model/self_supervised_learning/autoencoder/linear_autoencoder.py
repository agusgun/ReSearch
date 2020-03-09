import torch.nn as nn

from tqdm.notebook import tqdm
from .autoencoder import Autoencoder


class LinearAutoencoder(Autoencoder):
    '''
    Class for LinearAutoencoder using only linear layer and certain activation function
    '''
    def __init__(self, input_size, encoder=None, decoder=None):
        super(LinearAutoencoder, self).__init__()
        
        bottleneck_size = 32
        if not self.encoder:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, bottleneck_size)
            )
        if not self.decoder:
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_size),
                nn.Tanh()
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, train_dataloader, num_epochs, optimizer, criterion, print_loss=False):
        train_pbar = tqdm(range(num_epochs), desc='Train Progress')
        for epoch in train_pbar:
            for data in tqdm(train_dataloader, desc='Epoch: {}'.format(epoch+1), leave=False):
                x, _ = data
                x = x.view(x.size(0), -1)
                x = x.cuda()
                outputs = self.forward(x)
                loss = criterion(outputs, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.train_loss_history.append(loss.item())
            if print_loss:
                train_pbar.write('Epoch: {} \tTraining Loss: {:.6f}'.format(
                    epoch+1, 
                    loss.item()
                ))
            
    def predict(self, x):
        return self.forward(x)