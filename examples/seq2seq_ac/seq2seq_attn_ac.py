# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attentional Seq2seq with policy graident update.
"""

import argparse
import importlib

import torch
import torch.nn as nn

import texar.torch as tx
import os
import copy
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-model', type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_ac",
    help="The dataset config.")
args = parser.parse_args()

config_model = importlib.import_module(args.config_model)
config_data = importlib.import_module(args.config_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bleu(sampled_ids, target_ids, eos_token_id):
    """compute reward by r(y_{1:t}) = R(Y_{1:t}) - R(Y_{1:t-1})"""
    output_ids = sampled_ids.cpu()  # bsz, len
    target_ids = target_ids.cpu()  # bsz, len
    removed = [list(t) for t in target_ids]
    removed = [t[1:t.index(eos_token_id)] for t in removed]  # remove sos and eos
    # 怎么搞出 bleu 的 Tensor，就地操作是官方不推荐的 -> 改用 list + stack, 这个操作一般比较稳妥
    bleus = []
    # sampled_ids, dtype=torch.float32, requires_grad=True)
    for i in range(1, len(output_ids[0]) + 1):  # max_len
        partial = output_ids[:, :i]
        ps = []
        for hypo, ref in zip(partial, removed):
            sts_bleu = sentence_bleu([ref], hypo, auto_reweigh=True)
            ps.append(sts_bleu)  # partial sentence bleu
        if i == 1 or i == len(output_ids[0]):
            # first and last, set the difference
            bleus.append(torch.Tensor(ps))  # 32
        else:
            bleus.append(torch.Tensor(ps) - bleus[i - 2])  # Temporal difference
    bleus = torch.stack(bleus, dim=0)

    return bleus.transpose(1, 0).contiguous()


class Seq2SeqAttnActor(nn.Module):

    def __init__(self, train_data):
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
                beam_width=config_model.beam_width)

            return infer_outputs

        elif mode == 'pre-train-critic':  # rl training mode
            # with torch.no_grad():  # fix ?
            # we need the inner probability distribution of generated token
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)
            helper_infer = self.decoder.create_helper(
                decoding_strategy="infer_greedy",
                start_tokens=start_tokens,
                end_token=self.eos_token_id.item())
            infer_outputs, _, _, decoder_states = self.decoder(
                start_tokens=start_tokens,
                helper=helper_infer,
                end_token=self.eos_token_id.item(),
                memory=memory,
                memory_sequence_length=batch['source_length'],
                require_state=True,
                max_decoding_length=60)
            print("in bp4")

            # logits is contained in the outputs
            return infer_outputs, decoder_states


class Critic(nn.Module):
    """Critic model to guide the actor. Indeed, it is also a encoder-decoder.
    Q(s_t, a_t)
    """

    def __init__(self, train_data):
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
        # mapping from hidden output to vocab distribution
        # suppose actor and critic has same hidden size
        self.score_mapping = nn.Linear(self.decoder.cell.hidden_size * 2,
                                       train_data.target_vocab.size)
        self.tanh = nn.Tanh()
        self.mse_loss = nn.MSELoss()

    def forward(self, batch, sampled_id, reward=None, actor_states=None, mode='pre-train',
                target_actor=None, target_critic=None, regularization=True, lambda_c=1e-3):
        """
        input: generated token index
        output:  score distribution over vocabulary
        draft version: we only take the generated token into reward computation,
        """
        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch['source_text_ids']),
            sequence_length=batch['source_length'])

        memory = torch.cat(enc_outputs, dim=2)
        self.decoder.memory = memory
        #
        if mode == 'pre-train':  # train stage
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")
            sample_len = torch.Tensor([torch.sum(sent.ne(0)) for sent in sampled_id])

            outputs, _, _, states = self.decoder(
                memory=memory,
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=sampled_id,  # batch_size, len
                sequence_length=sample_len,
                require_state=True)
            # print(states)
            actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
            states = torch.stack([s.cell_state[0] for s in states], dim=0)  # len, bsz, hidden_size
            # print(states)
            # print(actor_states)
            # concat states : to take the decoder(actor) state into consideration
            concat_states = torch.cat((states, actor_states), dim=-1)  # len, bsz, hid_sz *2
            vocab_scores = self.score_mapping(concat_states)  # len, bsz, vocab_size
            seq_len, bsz, _ = vocab_scores.size()

            vocab_scores = self.tanh(vocab_scores).transpose(1, 0).contiguous()  # bsz, len, vocab_size
            predicted_scores = []
            for i in range(bsz):
                for j in range(seq_len):
                    predicted_scores.append(vocab_scores[i][j][sampled_id[i][j]])
            # print(index_to_select.size())

            # predicted_scores = torch.index_select(predicted_scores, 1, index_to_select)
            #     # predicted_scores.index_select(dim=1, index=index_to_select)  # bsz, len
            # print(predicted_scores.size())
            print(len(predicted_scores))

            predicted_scores = torch.stack(predicted_scores, dim=0).view(bsz, -1)
            # predicted_scores.requires_grad = True
            # qt = reward + p'(Y,1:t)  * q'
            actor_outputs, actor_states = target_actor(batch, mode='pre-train-critic')

            logits = actor_outputs.logits  # bsz, len, vocab_size

            prob = torch.softmax(logits, dim=-1)  # bsz, len, vocab_size

            q_score = target_critic(batch, sampled_id, mode='get_scores',
                                    actor_states=actor_states)  # bsz, len, vocab_size

            expectation = torch.sum(prob.detach() * q_score.detach(), dim=-1)  # bsz, len
            # seems that expectation and reward both have problem (detach() ? )
            qt = reward.detach() + expectation.detach()

            # print(predicted_scores.requires_grad)
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
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=sampled_id,
                sequence_length=sample_len,
                require_state=True)

            states = torch.stack([s.cell_state[0] for s in states], dim=0)  # len, bsz, hidden_size
            # concat states ? to get
            # print(states)
            # print(actor_states)

            actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
            concat_states = torch.cat((states, actor_states), dim=-1)  # len, bsz, hid_sz *2
            vocab_scores = self.score_mapping(concat_states)  # len, bsz, vocab_size
            vocab_scores = self.tanh(vocab_scores)  # len, bsz, vocab_size
            return vocab_scores.transpose(1, 0).contiguous()
        else:
            raise ValueError("Unsupported mode")

def _main():
    """pseudo-code"""

    train_data = tx.data.PairedTextData(
        hparams=config_data.train, device=device)
    val_data = tx.data.PairedTextData(
        hparams=config_data.val, device=device)
    test_data = tx.data.PairedTextData(
        hparams=config_data.test, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    actor = Seq2SeqAttnActor(train_data)
    critic = Critic(train_data)

    # actor.to(device)
    # critic.to(device)

    delay_actor = Seq2SeqAttnActor(train_data)
    delay_critic = Critic(train_data)

    for params in delay_critic.parameters():
        params.requires_grad = False
    for params in delay_actor.parameters():
        params.requires_grad = False

    mle_actor_train_op = tx.core.get_train_op(
        params=actor.parameters(), hparams=config_model.opt)

    pre_train_critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    rl_critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    rl_actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    def _mle_actor_train_epoch():
        data_iterator.switch_to_train_data()
        actor.train()

        step = 0
        for batch in data_iterator:
            loss = actor(batch, mode="pre-train")
            loss.backward()
            mle_actor_train_op()
            if step % config_data.display == 0:
                print("step={}, loss={:.4f}".format(step, loss))
            step += 1

    @torch.no_grad()
    def _actor_mle_eval_epoch(mode):
        if mode == 'val':
            data_iterator.switch_to_val_data()
        else:
            data_iterator.switch_to_test_data()
        actor.eval()

        refs, hypos = [], []
        for batch in data_iterator:
            infer_outputs = actor(batch, mode="mle_val")
            output_ids = infer_outputs["sample_id"][:, :, 0].cpu()
            target_texts_ori = [text[1:] for text in batch['target_text']]
            target_texts = tx.utils.strip_special_tokens(
                target_texts_ori, is_token_list=True)
            output_texts = tx.data.vocabulary.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            for hypo, ref in zip(output_texts, target_texts):
                hypos.append(hypo)
                refs.append([ref])

        return tx.evals.corpus_bleu_moses(
            list_of_references=refs, hypotheses=hypos)

    def pre_train_actor():
        best_val_bleu = -1.
        for i in range(config_data.pre_train_num_epochs):
            _mle_actor_train_epoch()

            val_bleu = _actor_mle_eval_epoch('val')
            best_val_bleu = max(best_val_bleu, val_bleu)
            print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                i, val_bleu, best_val_bleu))

            if val_bleu > best_val_bleu:
                model_path = os.path.join(args.output_dir, args.output_model)
                os.makedirs(model_path, exist_ok=True)
                print(f"Saving model to {model_path}")
                states = {
                    "model": actor.state_dict(),
                    # anything else?
                }
                torch.save(states, model_path)
            best_val_bleu = max(best_val_bleu, val_bleu)

            test_bleu = _actor_mle_eval_epoch('test')
            print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

            print('=' * 50)
        print("pre-train actor finished...")

    def pre_train_critic():
        step = 0
        for i in range(config_data.pre_train_num_epochs):
            data_iterator.switch_to_train_data()
            actor.eval()
            for batch in data_iterator:
                actor_outputs, actor_states = actor(batch, mode='pre-train-critic')
                # need to know what actor_outputs look lie
                logits = actor_outputs.logits  # bsz, len, vocab_size
                # print(logits.size())  # batch_size x max_len x vocab_size
                actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
                # critic_initial_state = critic.
                sampled_ids = torch.argmax(logits, dim=-1)  # bsz, len
                reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id)  # len_bsz
                # print("into critic")
                pre_train_critic_optimizer.zero_grad()

                loss = critic(batch, sampled_ids, reward=reward, target_actor=delay_actor,
                              target_critic=delay_critic, actor_states=actor_states)
                if step % config_data.display == 0:
                    print("pre-train loss at step {}: {}".format(step, loss))
                loss.backward()
                pre_train_critic_optimizer.step()  # run one optimizer step
                step += 1
                # assert 1 == 0

    def rl_training():
        print("start actor-critic training...")
        step = 0
        for i in range(config_data.rl_epochs):
            data_iterator.switch_to_train_data()
            for batch in data_iterator:
                mle_loss = actor(batch, mode="pre-train")
                actor_outputs, actor_states = actor(batch, mode='pre-train-critic')
                # need to know what actor_outputs look lie
                logits = actor_outputs.logits  # bsz, len, vocab_size
                # print(logits.size())  # batch_size x max_len x vocab_size
                # critic_initial_state = critic.
                sampled_ids = torch.argmax(logits, dim=-1)  # bsz, len
                reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id)  # len_bsz

                q_score = critic(batch, sampled_ids, mode='get_scores',
                                 actor_states=actor_states)  # bsz, len, vocab_size
                rl_loss = torch.sum(torch.sum(logits * q_score))
                total_loss = config_data.lambda_ll * mle_loss + rl_loss

                rl_actor_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                rl_actor_optimizer.step()

                critic_loss = critic(batch, sampled_ids, reward=reward, target_actor=delay_actor,
                                     target_critic=delay_critic, actor_states=actor_states)

                rl_critic_optimizer.zero_grad()
                critic_loss.backward()
                rl_critic_optimizer.step()
                # reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id)  #
                # len_bsz print("into critic")
                if step % config_data.display == 0:
                    print("actor critic step {} \n: actor: rl_loss: {} mle_loss {}\n"
                          "critic loss: {}".format(step, rl_loss, mle_loss, critic_loss))
                step += 1
                _delay_update_params()

    def _delay_update_params(initial=False):

        for d_p, p in zip(delay_critic.parameters(), critic.parameters()):
            d_p = config_data.fi_critic * p + d_p
        for d_p, p in zip(delay_actor.parameters(), actor.parameters()):
            d_p = config_data.fi_actor * p + d_p

    for d_p, p in zip(delay_critic.parameters(), critic.parameters()):
        d_p = p.clone()
    for d_p, p in zip(delay_actor.parameters(), actor.parameters()):
        d_p = p.clone()

    pre_train_actor()
    # copy params
    pre_train_critic()
    # copy  params
    rl_training()




if __name__ == '__main__':
    _main()
