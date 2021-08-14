import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from module.lstm_encoder import LSTMEncoder
from module.lstm_decoder import LSTMAttentionDecoder
from utils.beam_search import Beam
from utils.beam_search_pytorch import *
USE_CUDA = True


class Seq2SeqAttentionSharedEmbedding(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        attention_mode,
        batch_size,
        pad_token_src,
        pad_token_trg,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
        att_dropout=0,
        word2id=None,
        max_decode_len=None,
        id2word=None,
        input_feed=None,
    ):
        """Initialize model."""
        super(Seq2SeqAttentionSharedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        self.word2id = word2id
        self.max_decode_len = max_decode_len
        self.id2word = id2word
        # Assuming bi-directional LSTM
        self.encoder = LSTMEncoder(embedding_dim=emb_dim, hidden_dim=self.src_hidden_dim, vocab_size=vocab_size,
                                   encoder_dropout=dropout)

        self.decoder = LSTMAttentionDecoder(
            emb_dim,
            trg_hidden_dim,
            batch_first=True,
            att_dropout=att_dropout,
            input_feed=input_feed,
            # vocab_size=vocab_size
        )
        self.decoder.embeddings = self.encoder.embeddings
        self.encoder2decoder_scr_hm = nn.Linear(src_hidden_dim * 2, trg_hidden_dim, bias=False)
        self.encoder2decoder_ctx = nn.Linear(src_hidden_dim * 2, trg_hidden_dim, bias=False)

        self.decoder2vocab = nn.Linear(trg_hidden_dim, vocab_size, bias=False)

    def forward(self, input_src, input_src_len, input_trg):
        src_h, (_, _) = self.encoder(input_src, input_src_len)

        cur_batch_size = src_h.size()[0]
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(input_src_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((cur_batch_size, self.decoder.hidden_size)).cuda()

        ctx = self.encoder2decoder_ctx(src_h)

        #trg_emb = nn.Dropout(self.dropout)(self.encoder.embeddings(input_trg))
        trg_emb = self.encoder.embeddings(input_trg)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_h_0, decoder_c_0),
            ctx,
            input_src_len.view(-1)
        )

        decoder_logit = self.decoder2vocab(trg_h)
        return decoder_logit
    
    def load_encoder_embedding(self, emb, fix_emb=False):
        self.encoder.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        if fix_emb:
            self.encoder.embeddings.weight.requires_grad = False

    def load_word_embedding(self, id2word):
        import pickle
        emb = np.zeros((self.vocab_size, self.emb_dim))
        with open('feature/fasttextModel', 'br') as f:
            model = pickle.load(f)
        embed_dict = model.vocab

        for idx in range(self.vocab_size):
            word = id2word[idx]
            if word in embed_dict:
                vec = model.syn0[embed_dict[word].index]
                emb[idx] = vec
            else:
                if word == '<pad>':
                    emb[idx] = np.zeros([self.emb_dim])
                else:
                    emb[idx] = np.random.uniform(-1, 1, self.emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb))
        # self.word_embedding.weight.requires_grad = False

    def beam_decode_batch(self, input_src, input_src_len, beam_size=5):
        """Decode a minibatch."""
        # Get source minibatch

        #  (1) run the encoder on the src
        src_h, (_, _) = self.encoder(input_src, input_src_len)

        batch_size = src_h.size()[0]
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(input_src_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((batch_size, self.decoder.hidden_size)).cuda()

        ctx = self.encoder2decoder_ctx(src_h)

        beam = [
            Beam(beam_size, self.word2id, cuda=True)
            for _ in range(batch_size)
        ]

        ctx = ctx.data.repeat(beam_size, 1, 1)
        dec_states = [
            decoder_h_0.data.repeat(1, beam_size, 1),
            decoder_c_0.data.repeat(1, beam_size, 1)
        ]

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size

        for i in range(self.max_decode_len):
            next_input = torch.stack(
                [b.get_current_state() for b in beam if not b.done]
            ).t().contiguous().view(-1, 1)

            trg_emb = self.encoder.embeddings(next_input)
            trg_h, (trg_h_t, trg_c_t) = self.decoder(
                trg_emb,
                (dec_states[0].squeeze(0), dec_states[1].squeeze(0)),
                ctx,
                input_src_len.repeat(1, beam_size, 1).view(-1)
            )

            dec_states = (trg_h_t.unsqueeze(0), trg_c_t.unsqueeze(0))

            dec_out = trg_h_t.squeeze(1)
            out = torch.softmax(self.decoder2vocab(dec_out), dim=-1).unsqueeze(0)

            word_lk = out.view(
                beam_size,
                remaining_sents,
                -1
            ).transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx]):
                    active += [b]

                for dec_state in dec_states:  # iterate over h, c
                    # layers x beam*sent x dim
                    sent_states = dec_state.view(
                        -1, beam_size, remaining_sents, dec_state.size(2)
                    )[:, :, idx]
                    sent_states.data.copy_(
                        sent_states.data.index_select(
                            1,
                            beam[b].get_current_origin()
                        )
                    )

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = torch.cuda.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(
                    -1, remaining_sents,
                    self.decoder.hidden_size
                )
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) \
                    // remaining_sents
                return Variable(view.index_select(
                    1, active_idx
                ).view(*new_size))

            dec_states = (
                update_active(dec_states[0]),
                update_active(dec_states[1])
            )
            dec_out = update_active(dec_out)
            ctx = update_active(ctx)

            remaining_sents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = 1

        for b in range(batch_size):
            scores, ks = beam[b].sort_best()

            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]

        all_hyp_inds = [[x[0] for x in hyp] for hyp in allHyp]
        all_preds = [
            ' '.join([self.id2word[x.item()] for x in hyp])
            for hyp in all_hyp_inds
        ]

        return allHyp, allScores

    def greedy_decode_batch(self, input_src, input_src_len):
        """Decode a minibatch."""
        # Get source minibatch

        #  (1) run the encoder on the src
        src_h, (_, _) = self.encoder(input_src, input_src_len)

        batch_size = src_h.size()[0]
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(input_src_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((batch_size, self.decoder.hidden_size)).cuda()
        dec_states = (decoder_h_0, decoder_c_0)
        ctx = self.encoder2decoder_ctx(src_h)
        next_input = self.word2id["<s>"]
        next_input_tensor = torch.tensor([next_input] * batch_size).cuda()

        batched_ouput = []
        for step in range(self.max_decode_len):
            trg_emb = self.encoder.embeddings(next_input_tensor)

            trg_h, dec_states = self.decoder(
                trg_emb.unsqueeze(1),
                dec_states,
                ctx,
                input_src_len.view(-1)
            )

            decoder_logit = self.decoder2vocab(trg_h)

            greedy_next = torch.argmax(torch.softmax(decoder_logit, dim=-1), dim=-1)
            next_input_tensor = torch.tensor(greedy_next).cuda().view(-1)
            batched_ouput.append([self.id2word[token_id.item()] for token_id in greedy_next])
        return batched_ouput
    
    def beam_search(self, input_src, input_src_len):
        src_h, (_, _) = self.encoder(input_src, input_src_len)

        batch_size = src_h.size()[0]
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(input_src_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((batch_size, self.decoder.hidden_size)).cuda()
        dec_states = (decoder_h_0, decoder_c_0)
        ctx = self.encoder2decoder_ctx(src_h)
        next_input = self.word2id["<s>"]
        next_input_tensor = torch.tensor([next_input] * batch_size).cuda()

        batched_ouput = []
        for step in range(self.max_decode_len):
            trg_emb = self.encoder.embeddings(next_input_tensor)

            trg_h, dec_states = self.decoder(
                trg_emb.unsqueeze(1),
                dec_states,
                ctx,
                input_src_len.view(-1)
            )

            decoder_logit = self.decoder2vocab(trg_h)

            greedy_next = torch.argmax(torch.softmax(decoder_logit, dim=-1), dim=-1)
            next_input_tensor = torch.tensor(greedy_next).cuda().view(-1)
            batched_ouput.append([self.id2word[token_id.item()] for token_id in greedy_next])
        decoded_batch = beam_decode(input_src, dec_states, encoder_outputs=src_h)
        return decoded_batch
