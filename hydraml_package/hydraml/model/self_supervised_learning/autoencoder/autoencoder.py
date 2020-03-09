import torch.nn as nn


class Autoencoder(nn.Module):
    '''
    Abstract class for Autoencoder
    '''
    def __init__(self, encoder=None, decoder=None):
        super(Autoencoder, self).__init__()
        self.encoder, self.decoder = None, None
        if encoder != None: self.encoder = encoder
        if decoder != None: self.decoder = decoder
        self.train_loss_history = []
        
    def forward(self, x):
        raise NotImplementedError
        
    def fit(self, train_dataloader, num_epochs, optimizer, criterion, print_loss=False):
        raise NotImplementedError
        
    def predict(self, x):
        raise NotImplementedError
        
    def print_train_loss_history(self):
        for epoch, train_loss in enumerate(self.train_loss_history, 1):
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
            ))