import timm
import torch.nn as nn
import torchvision.models as models  # Para importar AlexNet

def get_model(model_name, num_classes=2, binary=False):
    """Cria modelo com head personalizado, suportando timm e torchvision"""
    if binary:
        num_classes = 1

    if model_name == 'alexnet':
        # Caso especial para AlexNet (via torchvision)
        model = models.alexnet(weights='DEFAULT')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        # Modelos do timm
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
    return model

def freeze_cnn_layers(model):
    """Congela camadas CNN """
    if model.__class__.__name__ == 'AlexNet':
        for param in model.features.parameters():
            param.requires_grad = False
        # Garante que a classifier está totalmente treinável
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Para modelos timm
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'get_classifier'):
            for param in model.get_classifier().parameters():
                param.requires_grad = True
    
    # Verificação final
    has_trainable = any(p.requires_grad for p in model.parameters())
    if not has_trainable:
        raise ValueError("Nenhum parâmetro está configurado para treinamento!")
    
    return model