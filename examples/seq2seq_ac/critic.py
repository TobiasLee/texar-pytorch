import torch.nn as nn
import torch
import texar.torch as tx


class Critic(nn.Module):
    """Critic model to guide the actor. Indeed, it is also a encoder-decoder.
    Q(s_t, a_t)
    note that we only need target information as auxiliary information
    """

    def __init__(self, train_data, config_model, config_data):
        super().__init__()
        # self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size
        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        #
        # self.source_embedder = tx.modules.WordEmbedder(
        #     vocab_size=self.source_vocab_size,
        #     hparams=config_model.embedder)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=config_model.embedder)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=self.target_embedder.dim,
            hparams=config_model.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=self.target_embedder.dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)
        # mapping from hidden output to vocab distribution
        # suppose actor and critic has same hidden size
        # self.score_mapping = nn.Linear(self.decoder.cell.hidden_size * 2,
        #                                train_data.target_vocab.size)
        self.score_mapping = nn.Linear(self.decoder.cell.hidden_size,
                                       train_data.target_vocab.size)
        # self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()
        self.max_decoding_len = config_data.max_decoding_len

    def forward(self, batch, sampled_id, logits=None, reward=None, actor_states=None, mode='pre-train',
                target_actor=None, target_critic=None, regularization=True, lambda_c=1e-3):
        """
        input: generated token index
        output:  score distribution over vocabulary
        """
        enc_outputs, _ = self.encoder(
            inputs=self.target_embedder(batch['target_text_ids']),
            sequence_length=batch['target_length'])

        memory = torch.cat(enc_outputs, dim=2)
        self.decoder.memory = memory
        #
        if mode == 'pre-train':  # train stage
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")
            sample_len = torch.Tensor([torch.sum(sent.ne(0)) for sent in sampled_id])

            outputs, _, _, states = self.decoder(
                memory=memory,
                memory_sequence_length=batch['target_length'],
                helper=helper_train,
                inputs=sampled_id,  # batch_size, len
                sequence_length=sample_len,
                require_state=True,
                max_decoding_length=self.max_decoding_len)
            # print(states)
            # actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
            states = torch.stack([s.cell_state[0] for s in states], dim=0)  # len, bsz, hidden_size
            # print(states)
            # concat_states = torch.cat((states, actor_states), dim=-1)  # len, bsz, hid_sz *2
            vocab_scores = self.score_mapping(states)  # len, bsz, vocab_size
            seq_len, bsz, _ = vocab_scores.size()

            vocab_scores = vocab_scores.transpose(1, 0).contiguous()  # bsz, len, vocab_size
            predicted_scores = []
            for i in range(bsz):
                for j in range(seq_len):
                    predicted_scores.append(vocab_scores[i][j][sampled_id[i][j]])

            predicted_scores = torch.stack(predicted_scores, dim=0).view(bsz, -1)
            # qt = reward + p'(Y,1:t)  * q'
            # actor_outputs, actor_states_t = target_actor(batch, mode='pre-train-critic')

            prob = torch.softmax(logits, dim=-1)  # bsz, len, vocab_size
            # actor_states_t = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size

            q_score = target_critic(batch, sampled_id, mode='get_scores',
                                    actor_states=actor_states)  # bsz, len, vocab_size

            expectation = torch.sum(prob * q_score, dim=-1)  # bsz, len
            # seems that expectation and reward both have problem (detach() ? )
            qt = reward.detach() + expectation.detach()
            print(torch.mean(torch.mean(predicted_scores, dim=1)))
            loss = self.mse_loss(qt, predicted_scores)
            if regularization:
                minus_average = predicted_scores - torch.mean(predicted_scores, dim=1, keepdim=True)
                loss += lambda_c * torch.sum(torch.mul(minus_average, minus_average))
            return loss

        elif mode == 'get_scores':  # used for evaluate V(s_{t+1})
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")
            sample_len = torch.Tensor([torch.sum(sent.ne(0)) for sent in sampled_id])
            outputs, _, _, states = self.decoder(
                memory=memory,
                memory_sequence_length=batch['target_length'],
                helper=helper_train,
                inputs=sampled_id,
                sequence_length=sample_len,
                require_state=True,
                max_decoding_length=self.max_decoding_len)

            states = torch.stack([s.cell_state[0] for s in states], dim=0)  # len, bsz, hidden_size
            # concat_states = torch.cat((states, actor_states), dim=-1)  # len, bsz, hid_sz *2
            vocab_scores = self.score_mapping(states)  # len, bsz, vocab_size
            # vocab_scores = self.tanh(vocab_scores)  # len, bsz, vocab_size
            return vocab_scores.transpose(1, 0).contiguous()
        else:
            raise ValueError("Unsupported mode")
