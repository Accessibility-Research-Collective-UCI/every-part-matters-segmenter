import torch

train_config = {
    'BATCH_SIZE': 4,
    'PROP_DROP': 0.1,
    'WINDOWS': 300,
    'HIDDEN_SIZE': 768,
    'EPOCH': 50,
    'LEARNING_RATE': 5e-5,
    'CRF_LEARNING_RATE': 1e-2,
    'LSTM_LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 0.01,
    'LR_WARMUP': 0.1,
    'MAX_GRAD_NORM': 1.0,
    'MAX_ENTITY_LEN': 10,
    'WIDTH_EMB_SIZE': 150,
    'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'SEED': 4
}