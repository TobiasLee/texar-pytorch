import torch.nn as nn
import torch
import texar.torch as tx


class Seq2SeqAttnActor(nn.Module):

    def __init__(self, train_data, config_model, config_data):
        super().__init__()

        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size

        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id

        self.source_embedder = tx.modules.WordEmbedder(
            vocab_size=self.source_vocab_size,
            hparams=config_model.embedder)

        self.target_embedder = tx.modules.WordEmbedder(
            vocab_size=self.target_vocab_size,
            hparams=config_model.embedder)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=self.source_embedder.dim,
            hparams=config_model.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=self.target_embedder.dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)
        self.beam_width = config_model.beam_width
        self.max_decoding_len = config_data.max_decoding_len

    def forward(self, batch, mode):
        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch['source_text_ids']),
            sequence_length=batch['source_length'])

        memory = torch.cat(enc_outputs, dim=2)

        if mode == "pre-train":  # use mle pre-train the actor
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")

            training_outputs, _, _ = self.decoder(
                memory=memory,
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=batch['target_text_ids'][:, :-1],
                sequence_length=batch['target_length'] - 1)

            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['target_text_ids'][:, 1:],
                logits=training_outputs.logits,
                sequence_length=batch['target_length'] - 1)

            return mle_loss
        elif mode == 'mle_eval':  # mle eval mode
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)
            infer_outputs = self.decoder(
                start_tokens=start_tokens,
                end_token=self.eos_token_id.item(),
                memory=memory,
                memory_sequence_length=batch['source_length'],
                beam_width=self.beam_width)

            return infer_outputs

        elif mode == 'pre-train-critic':  # rl training mode
            # with torch.no_grad():  # fix ?
            # we need the inner probability distribution of generated token
            # print("in actor pretrain")
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)
            helper_infer = self.decoder.create_helper(
                decoding_strategy="infer_greedy",
                start_tokens=start_tokens,
                end_token=self.eos_token_id.item())
            infer_outputs, _, _, decoder_states  = self.decoder(
                start_tokens=start_tokens,
                helper=helper_infer,
                end_token=self.eos_token_id.item(),
                memory=memory,
                memory_sequence_length=batch['source_length'],
                require_state=True,
                max_decoding_length=self.max_decoding_len)
            # logits is contained in the outputs
            return infer_outputs, decoder_states
        else:
            raise ValueError("unsupported mode")
