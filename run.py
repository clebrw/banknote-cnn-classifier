import torch
from config import *
from data import get_dataloaders
from models import get_model, freeze_cnn_layers
from train import train_model, evaluate
from utils import plot_metrics, plot_confusion_matrix
from early_stopping import EarlyStopping  # Nova importação


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    
    for model_name in MODELS_TO_TEST:
        print(f'\n=== Training {model_name} ===')
        
        # Prepara dados e modelo
        train_loader, val_loader, test_loader, classes = get_dataloaders(model_name)
        model = get_model(model_name, num_classes=len(classes)).to(device)
        model = freeze_cnn_layers(model)
        
        # Inicializa Early Stopping com caminho específico para cada modelo
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            verbose=True,
            path=f'checkpoint_{model_name}.pt'  # Nome único para cada modelo
        )
        # No seu run.py, após criar o modelo:
        print("\nParâmetros treináveis:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"- {name}")

        # Treina
        history = train_model(model, train_loader, val_loader, device, model_name, early_stopping)
        
        # Avalia
        test_loss, test_acc = evaluate(model, test_loader, device)
        test_acc /= len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        results[model_name] = {'test_acc': test_acc, 'test_loss': test_loss, 'history': history}
        
        # Plota resultados
        plot_metrics(history, model_name)
        plot_confusion_matrix(model, test_loader, classes, device, model_name)
    
    # Mostra resultados finais
    print('\n=== Resultados Finais ===')
    for name, res in results.items():
        print(f'{name}: {res["test_acc"]:.2%}')

if __name__ == '__main__':
    main()