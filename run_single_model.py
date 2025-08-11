#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para executar um único modelo específico
"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from config import *
from data import get_dataloaders
from models import get_model, freeze_cnn_layers
from train import train_model, evaluate_with_preds
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from early_stopping import EarlyStopping

def run_single_model(model_name, num_runs=1, learning_rate=None):
    """Executa um único modelo com configurações específicas"""
    os.makedirs('logs', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== Executando {model_name} em {device} ===\n")
    
    # Usar learning rate específico se fornecido
    lr = learning_rate if learning_rate is not None else LEARNING_RATE
    print(f"Learning rate: {lr:.0e}")
    
    all_histories = []
    
    for run in range(num_runs):
        print(f'\n=== Run {run+1}/{num_runs} para {model_name} ===\n')
        
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
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            device, 
            model_name, 
            seed, 
            early_stopping,
            learning_rate=lr
        )
        
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
        history['test_confusion_matrix'] = cm.tolist()

        all_histories.append(history)
    
    # Salvar resultados
    output_file = f'logs/{model_name}_{lr:.0e}_single.json'
    with open(output_file, 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    print(f"\nResultados salvos em {output_file}")
    
    # Calcular e exibir médias
    if num_runs > 1:
        avg_test_acc = sum(h['test_acc'] for h in all_histories) / num_runs
        avg_test_f1 = sum(h['test_f1'] for h in all_histories) / num_runs
        print(f"\nMédia de {num_runs} execuções:")
        print(f"Acurácia média: {avg_test_acc:.4f}")
        print(f"F1-score médio: {avg_test_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Executar um único modelo CNN')
    parser.add_argument('--model', type=str, required=True,
                        help='Nome do modelo para executar (ex: "resnet50", "vit_base_patch16_224")')
    parser.add_argument('--runs', type=int, default=1,
                        help='Número de execuções com diferentes seeds')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate específico (opcional)')
    
    args = parser.parse_args()
    
    run_single_model(args.model, num_runs=args.runs, learning_rate=args.lr)

if __name__ == '__main__':
    main()