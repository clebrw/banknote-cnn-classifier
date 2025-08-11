# Application of Convolutional Neural Networks in the Identification of Counterfeit Banknotes

This work presents a comparative study of convolutional neural network architectures applied to the detection of counterfeit banknotes using computer vision techniques. 

The main objective was to evaluate the performance of various CNN architectures in the task of image classification, using a dataset of real and counterfeit banknotes. The dataset consists of banknotes from Bangladesh, with high quality real samples from both origins (authentic and counterfeit).

## Modelos Disponíveis

O projeto agora suporta uma ampla variedade de arquiteturas de CNN, incluindo:

- **Modelos Clássicos**: AlexNet, ResNet50, DenseNet121
- **Modelos Eficientes**: EfficientNet-B0, MobileNetV3
- **Modelos Modernos**: ConvNeXt, EfficientNetV2
- **Transformers**: Vision Transformer (ViT), Swin Transformer

![bangladesh](imgs/dataset_bangladesh.png)

The methodology involved image preprocessing, data augmentation techniques, and model training with multiple random seeds to ensure statistical robustness. 

![AlexNet_seed_variation](imgs/accuracy_evolution_alexnet_1e-02.png)

Evaluation metrics included accuracy, standard deviation, and stability across runs.The results showed that AlexNet achieved the best overall performance, with test accuracy reaching up to 97.75% and high consistency between executions.

![all_seeds_AlexNet](imgs/alexnet_grafico_comparativo.png)

![Confusion_matrix](imgs/confusion_matrix_original_dataset.png)

ResNet-50 also performed well, although at a higher computational cost. DenseNet121 yielded consistent results, while EfficientNet-B0 underperformed, possibly due to its sensitivity to dataset size and hyperparameter configuration. 

It is concluded that simpler models such as AlexNet can be highly effective in specific classification tasks when properly tuned and combined with appropriate validation and data augmentation strategies.

## Novos Scripts

Foram adicionados novos scripts para facilitar o uso e teste de diferentes modelos:

- **list_models.py**: Lista os modelos disponíveis no timm com opções de filtragem e ordenação
  ```
  python list_models.py --filter vit --sort params
  ```

- **run_single_model.py**: Executa um único modelo específico com opções personalizadas
  ```
  python run_single_model.py --model vit_base_patch16_224 --runs 3 --lr 0.001
  ```

## Instalação

```
pip install -r requirements.txt
```
