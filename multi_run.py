import os
import json
import torch
import numpy as np    
from config import *
from pathlib import Path
from data import get_dataloaders
from models import get_model, freeze_cnn_layers
from train import train_model, evaluate_with_preds  # Usando a nova função
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from early_stopping import EarlyStopping

def run_multiple_times(model_name, num_runs=10):
    os.makedirs('logs', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    all_histories = []
    
    for run in range(num_runs):
        print(f'\n=== Run {run+1}/{num_runs} for {model_name} ===')
        
        # Fixar a semente para reprodutibilidade
        seed = SEED + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepara dados e modelo
        train_loader, val_loader, test_loader, classes = get_dataloaders(model_name)
        model = get_model(model_name, num_classes=len(classes), binary=False).to(device)
        model = freeze_cnn_layers(model)
        
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            path=f'logs/checkpoint_{model_name}_run{run}.pt'
        )
        
        # Treina
        history = train_model(model, train_loader, val_loader, device, model_name, seed, early_stopping)
        
        # Avaliação no conjunto de teste com métricas detalhadas
        print("→ Avaliando no conjunto de teste com métricas detalhadas...")
        test_loss, y_true, y_pred = evaluate_with_preds(model, test_loader, device)

        test_acc = (y_true == y_pred).mean()
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f'✓ Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
        print(f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}')
        print('Confusion Matrix:\n', cm)

        # Salvar métricas no histórico para salvar em JSON
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
        history['test_precision'] = precision
        history['test_recall'] = recall
        history['test_f1'] = f1
        history['test_confusion_matrix'] = cm.tolist()  # Convertendo para lista para salvar JSON

        all_histories.append(history)
    
    with open(f'logs/{model_name}_{LEARNING_RATE:.0e}.json', 'w') as f:
        json.dump(all_histories, f, indent=4)

if __name__ == '__main__':
    for model_name in MODELS_TO_TEST:
        run_multiple_times(model_name, num_runs=NUM_RUNS)
