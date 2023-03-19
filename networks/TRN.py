import math
import torch
import torch.nn as nn
import scipy.stats


# grabbed the cosine and sin similarity formulas from paper
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9391844
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    # what is max_len?
    def __init__(self, vocab_cnt=3048,d_model=22 * 25):
        super(PositionalEncoding, self).__init__()
        dropout = 0.1

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(vocab_cnt, d_model)
        position = torch.arange(0, vocab_cnt, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # 176000 67056
        print("x size",x.size())
        dropout_term = self.pe[:, : x.size(1), :]
        print("dropout_term size", dropout_term.size())
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x) # why is there dropout here 

class TRN(torch.nn.Module):
    """
    Neural Net with TRN
    """

    def __init__(self, batch_size, n_layers=2):
        super(TRN,self).__init__()
        num_classes = 4
        self.batch_size = batch_size
        self.d_model = 22 
        total_timesteps = 25

        self.positional_encoder = PositionalEncoding(d_model=self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=11, dropout=0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_layers) # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html 
        self.decoder = nn.Linear(self.d_model * total_timesteps, num_classes)

    def forward(self, x):
        # x = x.reshape(-1,1)
        # Avoid breaking if the last batch has a different size
        batch_size = x.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
       # x = self.positional_encoder(x)
       # binning of data from 250 to 25
        # print("x shape", x.size())
        # 32 * 22 * 250
        # print("x shape", x.size())

        avgpool = torch.nn.AvgPool1d(10)
        x = avgpool(x)

        x = x.reshape(self.batch_size, x.size(2), x.size(1))
        # normalize dadta
        x = self.encoder(x.float()) # * math.sqrt(self.d_model)
        x = x.reshape(self.batch_size, -1)
        x = self.decoder(x)
        return x
    
    def generate_square_subsequent_mask(sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
