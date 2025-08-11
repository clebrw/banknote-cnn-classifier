#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para listar modelos disponíveis no timm
"""

import argparse
import timm
from tabulate import tabulate

def list_models_simple(filter_pattern=None):
    """Lista modelos disponíveis no timm com filtragem simples"""
    # Obter lista de modelos pretrained disponíveis
    model_names = timm.list_models(pretrained=True)
    
    # Filtrar por padrão se fornecido
    if filter_pattern:
        model_names = [name for name in model_names if filter_pattern.lower() in name.lower()]
    
    return model_names

def main():
    parser = argparse.ArgumentParser(description='Listar modelos disponíveis no timm')
    parser.add_argument('--filter', type=str, default=None, 
                        help='Filtrar modelos por nome (ex: "resnet", "vit")')
    parser.add_argument('--top', type=int, default=20,
                        help='Número máximo de modelos para mostrar')
    
    args = parser.parse_args()
    
    print(f"\n=== Listando modelos disponíveis no timm ===\n")
    if args.filter:
        print(f"Filtrando por: {args.filter}")
    
    # Obter e imprimir modelos
    model_names = list_models_simple(filter_pattern=args.filter)
    
    # Limitar ao número máximo especificado
    if args.top and args.top < len(model_names):
        model_names = model_names[:args.top]
    
    # Imprimir tabela simples
    print(tabulate([[name] for name in model_names], headers=['Model Name'], tablefmt='pretty'))
    print(f"\nTotal de modelos: {len(model_names)}")
    
    # Sugestões para o usuário
    print("\nSugestões de modelos por categoria:")
    print("- CNNs clássicas: resnet50, densenet121")
    print("- CNNs eficientes: efficientnet_b0, mobilenetv3_large_100")
    print("- CNNs modernas: convnext_tiny, convnext_small")
    print("- Transformers: vit_base_patch16_224, swin_tiny_patch4_window7_224")
    
    print("\nPara usar um modelo, adicione-o à lista MODELS_TO_TEST em config.py")

if __name__ == '__main__':
    main()