import torch
from .torch_models import FullyConnected, LSTM, FullyConnectedTSL, LSTMTSL
import lightning as L

class LightningWrapper(L.LightningModule):
    def __init__(self, model, params):
        super().__init__()

        self.model = model
        self.lr = params['model']['params']['lr']
        self.save_hyperparameters(ignore=['model'])

    def forward(self, inputs, target):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)

        loss = torch.nn.functional.mse_loss(output.view(-1), target.view(-1))

        if hasattr(self.model, 'custom_loss'):
            loss += self.model.custom_loss

        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.mse_loss(output.view(-1), target.view(-1))
        self.log("val/loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.1)

def get_torch_model(parameters: dict, n_features_in, n_features_out) -> torch.nn.Module:
    
    selection_name = parameters['selection']['name']
    residual = parameters['selection'].get('params', dict()) or dict()
    residual = residual.get('residual', False)

    if selection_name == 'NoSelection':
        if parameters['model']['name'] == "dense":          
            model = LightningWrapper(FullyConnected(parameters, n_features_in=n_features_in, n_features_out=n_features_out), parameters)
        if parameters['model']['name'] == "lstm":          
            model = LightningWrapper(LSTM(parameters, n_features_in=n_features_in, n_features_out=n_features_out), parameters)
    elif 'TimeSelectionLayer' in selection_name:
        if parameters['model']['name'] == "dense":          
            model = LightningWrapper(FullyConnectedTSL(parameters, n_features_in=n_features_in, n_features_out=n_features_out), parameters)
        if parameters['model']['name'] == "lstm":          
            model = LightningWrapper(LSTMTSL(parameters, n_features_in=n_features_in, n_features_out=n_features_out), parameters)
    else:
        raise NotImplementedError()

    return model