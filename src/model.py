import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SoccerNetTransformer(nn.Module):
    def __init__(
        self,
        input_dim=512,
        d_model=384,
        num_heads=4,
        num_layers=3,
        dim_feedforward=768,
        dropout=0.1,
        num_classes=18
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def get_encoder_output(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        """
        Single TCN block with dilated convolution, layer norm,
        ReLU activation and residual connection.

        d_model     : number of channels (feature dimension)
        kernel_size : convolution kernel size
        dilation    : dilation factor for the convolution
        dropout     : dropout probability
        """
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x shape: (batch, seq_len, d_model)
        """
        residual = x
        out = x.transpose(1, 2)
        out = self.conv(out)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class SoccerNetTCN(nn.Module):
    def __init__(
        self,
        input_dim=512,
        d_model=256,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        num_classes=18
    ):
        """
        Temporal Convolutional Network for dense action spotting.
        Processes full match halves and predicts at every timestamp.

        input_dim  : input feature dimension (512 for PCA features)
        d_model    : internal channel dimension
        num_layers : number of TCN blocks (each doubles dilation)
        kernel_size: convolution kernel size
        dropout    : dropout probability
        num_classes: number of output classes including background
        """
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.tcn_blocks = nn.ModuleList([
            TCNBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(num_layers)
        ])

        self.output_projection = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        returns: (batch, seq_len, num_classes) logits at every timestamp
        """
        x = self.input_projection(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.output_projection(x)
        return x

    def get_encoder_output(self, x):
        """
        Returns encoder output before classification head.
        Used for Stage 1 MFM pretraining.
        shape: (batch, seq_len, d_model)
        """
        x = self.input_projection(x)
        for block in self.tcn_blocks:
            x = block(x)
        return x