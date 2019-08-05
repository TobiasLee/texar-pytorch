pre_train_num_epochs = 15
rl_epochs = 10


display = 50
lambda_ll = 0.1
source_vocab_file = './data/iwslt14/vocab.de'
target_vocab_file = './data/iwslt14/vocab.en'

fi_critic = 1e-3
fi_actor = 1e-3

train = {
    'batch_size': 64,
    'allow_smaller_final_batch': False,
    'source_dataset': {
        "files": 'data/iwslt14/train.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/train.en',
        'vocab_file': target_vocab_file,
    }
}

val = {
    'batch_size': 64,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/valid.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/valid.en',
        'vocab_file': target_vocab_file,
    }
}

test = {
    'batch_size': 64,
    'shuffle': False,
    'source_dataset': {
        "files": 'data/iwslt14/test.de',
        'vocab_file': source_vocab_file,
    },
    'target_dataset': {
        'files': 'data/iwslt14/test.en',
        'vocab_file': target_vocab_file,
    }
}
