import timm
import torch.nn as nn
import torchvision.models as models  # Para importar AlexNet
import pandas as pd
from tabulate import tabulate

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

def list_available_models(filter_pattern=None, sort_by='params', ascending=True, top_n=20):
    """Lista modelos disponíveis no timm com opções de filtragem e ordenação
    
    Args:
        filter_pattern (str, optional): Padrão para filtrar nomes de modelos. Default: None
        sort_by (str, optional): Campo para ordenação ('params', 'gflops', 'year'). Default: 'params'
        ascending (bool, optional): Ordenar em ordem crescente. Default: True
        top_n (int, optional): Número máximo de modelos para mostrar. Default: 20
        
    Returns:
        DataFrame: DataFrame com informações dos modelos
    """
    # Obter lista de modelos pretrained disponíveis
    model_names = timm.list_models(pretrained=True)
    
    # Filtrar por padrão se fornecido
    if filter_pattern:
        model_names = [name for name in model_names if filter_pattern.lower() in name.lower()]
    
    # Coletar informações sobre cada modelo
    models_info = []
    for name in model_names:
        try:
            # Criar modelo temporário para obter informações
            model = timm.create_model(name, pretrained=False)
            num_params = sum(p.numel() for p in model.parameters()) / 1e6  # em milhões
            
            # Extrair informações de arquitetura
            architecture_type = 'CNN'  # padrão
            if 'vit' in name.lower() or 'deit' in name.lower():
                architecture_type = 'Vision Transformer'
            elif 'swin' in name.lower():
                architecture_type = 'Swin Transformer'
            elif 'convnext' in name.lower():
                architecture_type = 'ConvNeXt'
            elif 'efficientnet' in name.lower():
                architecture_type = 'EfficientNet'
            elif 'resnet' in name.lower() or 'resnext' in name.lower():
                architecture_type = 'ResNet/ResNeXt'
            elif 'densenet' in name.lower():
                architecture_type = 'DenseNet'
            elif 'mobilenet' in name.lower():
                architecture_type = 'MobileNet'
            
            # Estimar ano de lançamento (aproximado)
            year = 2015  # padrão
            if 'vit' in name.lower() or 'deit' in name.lower():
                year = 2020
            elif 'swin' in name.lower():
                year = 2021
            elif 'convnext' in name.lower():
                year = 2022
            elif 'efficientnetv2' in name.lower():
                year = 2021
            elif 'efficientnet' in name.lower():
                year = 2019
            elif 'mobilenetv3' in name.lower():
                year = 2019
            elif 'mobilenetv2' in name.lower():
                year = 2018
            
            models_info.append({
                'name': name,
                'architecture': architecture_type,
                'params': num_params,
                'year': year
            })
        except Exception as e:
            # Ignorar modelos que não podem ser carregados
            continue
    
    # Converter para DataFrame e ordenar
    df = pd.DataFrame(models_info)
    if not df.empty:
        df = df.sort_values(by=sort_by, ascending=ascending).head(top_n)
    
    return df

def print_models_table(df):
    """Imprime tabela formatada de modelos"""
    if df.empty:
        print("Nenhum modelo encontrado com os critérios especificados.")
        return
    
    # Formatar parâmetros para melhor legibilidade
    df['params'] = df['params'].apply(lambda x: f"{x:.2f}M")
    
    # Imprimir tabela formatada
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    print(f"\nTotal de modelos: {len(df)}")

def freeze_cnn_layers(model):
    """Congela camadas de extração de características, mantendo apenas a cabeça de classificação treinável"""
    model_name = model.__class__.__name__
    
    # Caso específico para AlexNet do torchvision
    if model_name == 'AlexNet':
        for param in model.features.parameters():
            param.requires_grad = False
        # Garante que a classifier está totalmente treinável
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Caso específico para Vision Transformers (ViT)
    elif 'ViT' in model_name or model_name.startswith('Swin'):
        # Congela os blocos do transformer, mantendo apenas a cabeça de classificação treinável
        if hasattr(model, 'blocks'):
            for param in model.patch_embed.parameters():
                param.requires_grad = False
            for param in model.blocks.parameters():
                param.requires_grad = False
            # Mantém a cabeça de classificação treinável
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
        else:
            # Abordagem genérica para outros modelos baseados em transformer
            for name, param in model.named_parameters():
                if 'head' not in name and 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
    
    # Caso específico para ConvNeXt
    elif 'ConvNeXt' in model_name:
        # Congela os estágios de features, mantendo apenas a cabeça treinável
        if hasattr(model, 'stages'):
            for param in model.stem.parameters():
                param.requires_grad = False
            for param in model.stages.parameters():
                param.requires_grad = False
            # Mantém a cabeça de classificação treinável
            if hasattr(model, 'head'):
                for param in model.head.parameters():
                    param.requires_grad = True
    
    # Abordagem genérica para outros modelos do timm
    else:
        # Para modelos timm
        for param in model.parameters():
            param.requires_grad = False
        
        # Tenta diferentes atributos de classificador comuns
        if hasattr(model, 'get_classifier') and callable(getattr(model, 'get_classifier')):
            for param in model.get_classifier().parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
    
    # Verificação final
    has_trainable = any(p.requires_grad for p in model.parameters())
    if not has_trainable:
        raise ValueError("Nenhum parâmetro está configurado para treinamento!")
    
    return model