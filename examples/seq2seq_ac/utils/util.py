from nltk.translate.bleu_score import sentence_bleu
import torch


def compute_bleu(sampled_ids, target_ids, eos_token_id, device):
    """compute reward by r(y_{1:t}) = R(Y_{1:t}) - R(Y_{1:t-1})"""
    output_ids = sampled_ids.cpu()  # bsz, len
    target_ids = target_ids.cpu()  # bsz, len
    removed = [list(t) for t in target_ids]
    removed = [t[1:t.index(eos_token_id)] for t in removed]  # remove sos and eos
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

    return bleus.transpose(1, 0).contiguous().to(device)
