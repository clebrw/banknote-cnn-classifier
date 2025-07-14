import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

def plot_metrics(history, model_name):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.suptitle(model_name)
    plt.savefig(f'outputs/{model_name}_metrics.png')
    plt.close()

def plot_confusion_matrix(model, loader, classes, device, model_name):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    plt.savefig(f'outputs/{model_name}_cm.png')
    plt.close()