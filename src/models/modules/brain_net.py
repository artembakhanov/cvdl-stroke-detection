import torch
from torch import nn

from src.models.modules.sequence import SequenceConv2d, SequenceFSM
from src.utils.utils import calc_embedding_size_recursive, batch_to_tensor


class BrainNet(nn.Module):
    def __init__(self,
                 hparams: dict
                 ):
        super().__init__()
        input_size = hparams["input_size"]
        if type(input_size) == int:
            input_size = (input_size, input_size)
        sequence_size = hparams["sequence_size"]
        hidden_dim = hparams["hidden_dim"]
        n_lstm_layers = hparams["n_lstm_layers"]
        self.sequence_convolutions = nn.Sequential(
            SequenceConv2d(1, 16, 11, 2, 0, sequence_size),
            SequenceConv2d(16, 32, 5, 2, 1, sequence_size),
            SequenceConv2d(32, 48, 5, 2, 0, sequence_size)
        )
        first_convs = list(map(lambda seq_conv: seq_conv.convs[0], self.sequence_convolutions))
        emb_dim = calc_embedding_size_recursive(first_convs, input_size)
        emb_dim = 48 * emb_dim[0] * emb_dim[1]  # last out_channels * emb_dim * emb_dim
        self.fsms = SequenceFSM(48, sequence_size)

        lstm_dropout = 0.3 if n_lstm_layers > 1 else 0
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_lstm_layers, dropout=lstm_dropout, batch_first=True,
                            bidirectional=True)

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequences):
        sequences = list(map(lambda sequence: self.sequence_convolutions(sequence), sequences))
        sequences = batch_to_tensor(sequences, included_y=False).transpose(1, 0)
        sequences = list(map(lambda sequence: self.fsms(sequence), sequences))
        sequences = batch_to_tensor(sequences, included_y=False).transpose(1, 0)
        sequences = torch.flatten(sequences, start_dim=2, end_dim=-1)  # (batch_size, sequence_length, emb_dim)
        lstm_out, (ht, ct) = self.lstm(sequences)
        out = self.dense(ht[-1])
        return out
