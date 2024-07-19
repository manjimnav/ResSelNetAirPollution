import torch
from torch import nn
from .initializer import init_weights
from .layer import TimeSelectionLayer

class FullyConnectedTSL(nn.Module):
    def __init__(self, parameters: dict,  n_features_in: int, n_features_out: int):
        super().__init__()

        n_layers = parameters['model']['params']['layers']
        n_units = parameters['model']['params']['units']
        self.dropout = parameters['model']['params']['dropout']
        self.pred_len = parameters['dataset']['params']['pred_len']
        self.seq_len = parameters['dataset']['params']['seq_len']
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.residual = parameters['selection']["params"].get("residual", False)

        self.module_list = []
        
        layer_input_size = self.seq_len*n_features_in
        for _ in range(n_layers):
            self.module_list.extend([nn.Linear(layer_input_size, n_units)])
            layer_input_size = n_units

        self.output_layer = nn.Linear(n_units, self.n_features_out*self.pred_len)
        self.module_list = nn.ModuleList(self.module_list)

        if self.residual:
            self.input_tsl = TimeSelectionLayer((self.seq_len*n_features_in,), self.n_features_out*self.pred_len, parameters['selection']["params"]["regularization"])
            self.tsl_residuals = nn.ModuleList([TimeSelectionLayer((self.seq_len*n_features_in,), self.n_features_out*self.pred_len, parameters['selection']["params"]["regularization"]) for _ in range(n_layers)])
        else:
            self.input_tsl = TimeSelectionLayer((self.seq_len*n_features_in,), self.n_features_out*self.pred_len, parameters['selection']["params"]["regularization"])

    def compute_custom_loss(self):

        self.custom_loss = torch.tensor(0)

        self.custom_loss += self.input_tsl.regularization_loss()

        if self.residual:
            for layer in self.tsl_residuals:
                self.custom_loss += layer.regularization_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # output of shape batch_size x n_timesteps_out
        x = x.reshape(x.shape[0], -1)
        original_inputs = torch.tensor(x)
        x = self.input_tsl(x)

        for indx, layer in enumerate(self.module_list):
            x = layer(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(self.dropout)(x)

            if self.residual:
                x = torch.concatenate((x, self.tsl_residuals[indx](original_inputs)), axis=-1)

        self.compute_custom_loss()

        return self.output_layer(x).reshape(x.shape[0], self.pred_len, self.n_features_out)