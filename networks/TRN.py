import math
import torch
import torch.nn as nn


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
    def __init__(self, vocab_cnt=3048,d_model=22 * 250):
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

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TRN(torch.nn.Module):
    """
    Neural Net with TRN
    """

    def __init__(self, batch_size, n_layers=10):
        super(TRN,self).__init__()
        num_classes = 4
        self.d_model = 22 * 250
        # self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)

       # self.flatten = nn.Flatten()
        self.positional_encoder = PositionalEncoding(d_model=self.d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=22 )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=12)

        self.encoder = nn.Embedding(num_classes,self.d_model)
        self.decoder = nn.Linear(self.d_model,num_classes)
        self.init_weights()
        # self.sigmoid = nn.Sigmoid()
        
    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # src = self.encoder(src)
        # src = self.positional_encoder(src)

        # output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output)
        # return output
        print("x type",type(x))
        x = self.encoder(x.long()) * math.sqrt(self.d_model)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)
        x= x.mean(dim=1)
        x = self.decoder(x)
        return x
    

    
    def generate_square_subsequent_mask(sz):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
