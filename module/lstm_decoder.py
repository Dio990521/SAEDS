import torch
import torch.nn as nn
from module.global_attention import GlobalAttention


class LSTMAttentionDecoder(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, att_dropout=0, input_feed=False, vocab_size=None):
        """Initialize params."""
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.input_feed = input_feed
        if input_feed:
            input_size = input_size + hidden_size

        self.decoding_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

        self.attention_layer = GlobalAttention(hidden_size, dropout=att_dropout)
        if vocab_size is not None:
            self.embeddings = nn.Embedding(vocab_size, input_size, padding_idx=0)

    def forward(self, trg_emb, hidden, ctx, src_len):
        def recurrence(_trg_emb_i, _hidden, _h_tilde):
            if self.input_feed:
                _lstm_input = torch.cat((_trg_emb_i, _h_tilde.squeeze(0)), dim=1)
            else:
                _lstm_input = _trg_emb_i
            lstm_out, _hidden = self.decoding_lstm(_lstm_input.unsqueeze(1), _hidden)
            _h_tilde, _ = self.attention_layer(lstm_out, ctx, src_len)

            return _h_tilde.squeeze(0), _hidden # squeeze out the trg_len dimension

        output = []
        if len(hidden[0].size()) == 2:
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        h_tilde = hidden[0]
        for i in range(trg_emb.size()[1]):
            trg_emb_i = trg_emb[:, i, :]
            h_tilde, hidden = recurrence(trg_emb_i, hidden, h_tilde)
            output.append(h_tilde)

        output = torch.stack(output, dim=0).transpose(0, 1)

        return output, hidden
