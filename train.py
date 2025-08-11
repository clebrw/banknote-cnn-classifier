import torch
import time
from tqdm import tqdm
from pathlib import Path
from config import *
import numpy as np

def train_model(model, train_loader, val_loader, device, model_name, seed, early_stopping=None, learning_rate=None):
    criterion = torch.nn.CrossEntropyLoss()
    # Usar learning rate específico se fornecido, caso contrário usar o padrão da config
    lr = learning_rate if learning_rate is not None else LEARNING_RATE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Adicionando o scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE,
    )
    epoch_logs = []
    best_acc = 0
    
    # Garante que o diretório de output existe
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        start_time = time.time()
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        epoch_time = time.time() - start_time

        # Validação
        val_loss, val_correct = evaluate(model, val_loader, device, criterion, normalize=False)
        
        # Calcula métricas
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Atualiza o scheduler com a loss de validação
        scheduler.step(val_loss)
        
        # Registra a taxa de aprendizado atual
        current_lr = optimizer.param_groups[0]['lr']

        # Log de época
        epoch_logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_time": epoch_time
        })
        
        # Log de progresso
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} - {epoch_time:.2f}s')
        print(f'LR: {current_lr:.2e}')  # Mostra a taxa de aprendizado atual
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
        
        # Salva melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = OUTPUT_DIR/f'{model_name}_best_{LEARNING_RATE}.pth'
            torch.save(model.state_dict(), best_path)
            print(f'Model improved! Saved to {best_path}')
        
        # Early Stopping
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    
    return {
        "seed": seed,
        "epochs": epoch_logs
    }


def evaluate(model, loader, device, criterion=None, normalize=True):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
        
    model.eval()
    loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    if normalize:
        return loss / total, correct / total
    else:
        return loss, correct


def evaluate_with_preds(model, loader, device, criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
        
    model.eval()
    loss, total = 0.0, 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            total += labels.size(0)
    
    avg_loss = loss / total
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return avg_loss, y_true, y_pred
