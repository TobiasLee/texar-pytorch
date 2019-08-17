from nltk.translate.bleu_score import sentence_bleu as nltk_bleu
from nltk.translate.bleu_score import SmoothingFunction

import torch
import numpy as np

chencherry = SmoothingFunction()


def compute_bleu(sampled_ids, target_ids, eos_token_id, device):
    """compute reward by r(y_{1:t}) = R(Y_{1:t}) - R(Y_{1:t-1})"""
    output_ids = sampled_ids.cpu()  # bsz, len
    target_ids = target_ids.cpu()  # bsz, len
    removed = [t.numpy() for t in target_ids]
    # print(np.where(removed[0] == eos_token_id)[0][0])
    removed = [t[1:np.where(t == eos_token_id)[0][0]] for t in removed]  # remove sos and eos
    bleus = []
    # sampled_ids, dtype=torch.float32, requires_grad=True)
    for i in range(1, len(output_ids[0]) + 1):  # max_len
        partial = output_ids[:, :i]
        ps = []
        for hypo, ref in zip(partial, removed):
            hypo = hypo.numpy()

            sts_bleu = nltk_bleu([ref], hypo, auto_reweigh=True, smoothing_function=chencherry.method4)
            ps.append(sts_bleu)  # partial sentence bleu

        if i == 1:
            # first and last, set the difference
            bleus.append(torch.Tensor(ps))  # 32
        else:
            bleus.append(torch.Tensor(ps) - bleus[i - 2])  # Temporal difference
    bleus = torch.stack(bleus, dim=0)

    return bleus.transpose(1, 0).contiguous().to(device)


if __name__ == '__main__':
    target = torch.Tensor([[0, 1, 1, 2, 2, 4], [0, 1, 2, 3, 2, 4]])
    output = torch.Tensor([[1, 13, 2, 2], [1, 2, 3, 2]])
    print(compute_bleu(sampled_ids=output, target_ids=target, eos_token_id=4, device='cpu'))
