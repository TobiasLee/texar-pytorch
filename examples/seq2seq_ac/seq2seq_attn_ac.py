"""Attentional Seq2seq with actor-critic rl training."""

import argparse
import importlib

import torch

import texar.torch as tx
import os
from tqdm import tqdm
from utils.util import compute_bleu
from actor import Seq2SeqAttnActor
from critic import Critic
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config-model', type=str, default="config_model",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_ac",
    help="The dataset config.")
# parser.add_argument(
#     '--output-model', type=str, default="actor",
#     help="The model name of pre-trained actor.")
parser.add_argument(
    '--output-dir', type=str, default="./models/",
    help="The dataset config.")
parser.add_argument(
    '--lambda_ll', type=float, default=1.0,
    help="The coefficient of likelihood training loss.")
parser.add_argument(
    '--lambda_rl', type=float, default=1.0,
    help="The coefficient of rl training loss.")

args = parser.parse_args()

config_model = importlib.import_module(args.config_model)
config_data = importlib.import_module(args.config_data)

config_data.lambda_ll = args.lambda_ll
config_data.lambda_rl = args.lambda_rl
print("mle: {} ll : {}".format(config_data.lambda_ll, config_data.lambda_rl))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    actor = Seq2SeqAttnActor(train_data, config_model, config_data)
    critic = Critic(train_data, config_model, config_data)

    delay_actor = Seq2SeqAttnActor(train_data, config_model, config_data)
    delay_critic = Critic(train_data, config_model, config_data)

    actor.to(device)
    critic.to(device)
    delay_actor.to(device)
    delay_critic.to(device)

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
        step = 1
        total_loss = 0.0
        t = tqdm(data_iterator.get_train_iterator())
        for batch in t:
            loss = actor(batch, mode="pre-train")
            loss.backward()
            mle_actor_train_op()
            total_loss += loss
            t.set_description("step={}, avg loss={:.4f}".format(step, total_loss / step))
            # if step % config_data.display == 0:
            #     print("step={}, loss={:.4f}".format(step, loss))
            step += 1

    @torch.no_grad()
    def _actor_mle_eval_epoch(mode):
        if mode == 'val':
            data_iterator.switch_to_val_data()
        else:
            data_iterator.switch_to_test_data()
        actor.eval()
        print("start evaluating....")
        refs, hypos = [], []
        for batch in data_iterator:
            infer_outputs = actor(batch, mode="mle_eval")
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
        for i in range(config_data.pre_train_actor_epochs):
            _mle_actor_train_epoch()

            val_bleu = _actor_mle_eval_epoch('val')
            print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                i, val_bleu, best_val_bleu))

            if val_bleu > best_val_bleu:
                best_val_bleu = max(best_val_bleu, val_bleu)
                model_path = os.path.join(args.output_dir, "actor-pre-train-best")
                os.makedirs(model_path, exist_ok=True)
                print(f"Saving model to {model_path}")
                states = {
                    "model": actor.state_dict(),
                    # anything else?
                }
                torch.save(states, model_path + "/model.ckpt")
            best_val_bleu = max(best_val_bleu, val_bleu)

            test_bleu = _actor_mle_eval_epoch('test')
            print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

            print('=' * 50)
        print("pre-train actor finished...")

    def pre_train_critic():
        step = 1
        total_loss = 0.0
        for i in range(config_data.pre_train_critic_epochs):
            data_iterator.switch_to_train_data()
            actor.eval()
            t = tqdm(data_iterator.get_train_iterator())

            for batch in t:
                actor_outputs, actor_states = delay_actor(batch, mode='pre-train-critic')
                # need to know what actor_outputs look lie
                logits = actor_outputs.logits  # bsz, len, vocab_size
                # print(logits.size())  # batch_size x max_len x vocab_size
                actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
                # critic_initial_state = critic.
                sampled_ids = torch.argmax(logits, dim=-1)  # bsz, len
                t1 = time.time()
                reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id, device=device)  # len_bsz
                bleu_time = time.time() - t1
                # print("into critic")
                pre_train_critic_optimizer.zero_grad()

                loss = critic(batch, sampled_ids, logits=logits, reward=reward, target_actor=delay_actor,
                              target_critic=delay_critic, actor_states=actor_states)
                total_loss += loss
                t.set_description("step={}, avg loss={:.4f}, compute bleu cost: {}".format(step, total_loss / step, bleu_time))
                #    if step % config_data.display == 0:
                #        print("pre-train loss at step {}: {}".format(step, loss))
                loss.backward()
                pre_train_critic_optimizer.step()  # run one optimizer step
                step += 1
                if step % config_data.save_interval == 0:
                    model_path = os.path.join(args.output_dir, "critic-pre-train")
                    os.makedirs(model_path, exist_ok=True)
                    print(f"Saving model to {model_path}")
                    torch.save(critic.state_dict(), model_path + "/critic-{}.ckpt".format(step))
                # update target critic
                for d_p, p in zip(delay_critic.parameters(), critic.parameters()):
                    d_p.data.copy_(config_data.fi_critic * p.data + d_p.data)

        print(f"Pretrain finished, saving model")
        torch.save(critic.state_dict(), model_path + "/critic-final.ckpt")

    def rl_training():
        print("start actor-critic training...")
        step = 1
        actor.train()
        critic.train()
        for i in range(config_data.rl_epochs):
            data_iterator.switch_to_train_data()
            for batch in data_iterator:
                #### actor step ####
                mle_loss = actor(batch, mode="pre-train")
                actor_outputs, actor_states = actor(batch, mode='pre-train-critic')
                logits = actor_outputs.logits  # bsz, len, vocab_size
                sampled_ids = torch.argmax(logits, dim=-1)  # bsz, len
                actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
                q_score = critic(batch, sampled_ids, mode='get_scores',
                                 actor_states=actor_states)  # bsz, len, vocab_size
                # need mask ?
                rl_loss = - torch.sum(torch.mean(logits * q_score))
                total_loss = config_data.lambda_ll * mle_loss + config_data.lambda_rl * rl_loss

                rl_actor_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                rl_actor_optimizer.step()

                #### critic step ####
                actor_outputs, actor_states = delay_actor(batch, mode='pre-train-critic')
                logits = actor_outputs.logits  # bsz, len, vocab_size
                actor_states = torch.stack([s.cell_state[0] for s in actor_states], dim=0)  # len, bsz, hidden_size
                sampled_ids = torch.argmax(logits, dim=-1)  # bsz, len
                reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id, device=device)  # len_bsz
                rl_critic_optimizer.zero_grad()

                critic_loss = critic(batch, sampled_ids, logits=logits, reward=reward, target_actor=delay_actor,
                                     target_critic=delay_critic, actor_states=actor_states)
                critic_loss.backward()
                rl_critic_optimizer.step()
                # reward = compute_bleu(sampled_ids, batch['target_text_ids'], eos_token_id=actor.eos_token_id)  #
                # len_bsz print("into critic")
                if step % config_data.display == 0:
                    print("actor critic step {}: \n actor: rl_loss: {} mle_loss {}\n"
                          "critic loss: {}".format(step, rl_loss, mle_loss, critic_loss))

                if step % config_data.eval_interval == 0:
                    print("evaluate at step: {} bleu: {}".format(step, _actor_mle_eval_epoch('val')))
                step += 1
                _delay_update_params()

    def _delay_update_params():
        for d_p, p in zip(delay_critic.parameters(), critic.parameters()):
            d_p.data.copy_(config_data.fi_critic * p.data + d_p.data)
        for d_p, p in zip(delay_actor.parameters(), actor.parameters()):
            d_p.data.copy_(config_data.fi_actor * p.data + d_p.data)

    print("loading pre-trained actor")
    actor.load_state_dict(torch.load("./models/actor-pre-train-best/model.ckpt")["model"])
    print("load successfully!")
    #    pre_train_actor()
    # copy params
#    print("loaded bleu on val ", _actor_mle_eval_epoch('val'))
    print("loading pre-trained critic")
    critic.load_state_dict(torch.load("./models/critic-pre-train/critic-final.ckpt"))
    print("load successfully!")

    for d_p, p in zip(delay_critic.parameters(), critic.parameters()):
        d_p.data.copy_(p.data)
    for d_p, p in zip(delay_actor.parameters(), actor.parameters()):
        d_p.data.copy_(p.data)


    # pre_train_critic()
    rl_training()


if __name__ == '__main__':
    _main()
