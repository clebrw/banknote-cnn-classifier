import os
from pathlib import Path

# Configurações de caminhos
DATA_DIR = Path('/Users/clebrw/projetos/Datasets/banknote/')
TRAIN_PATH = DATA_DIR / 'train'
# TRAIN_PATH = DATA_DIR / 'train_aug'
VAL_PATH = DATA_DIR / 'val'
TEST_PATH = DATA_DIR / 'test'
OUTPUT_DIR = Path('outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurações de treino
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_PATIENCE = 3
SEED = 42
NUM_RUNS = 10

# Lista de modelos para testar
MODELS_TO_TEST = ['alexnet','resnet50','densenet121','efficientnet_b0']