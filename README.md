# colovo

## Fluxo de Preparação e Treinamento da Segmentação (Auto-Labeling + U-Net)
Este diagrama detalha o processo desde as imagens brutas em formato de câmera até a exportação do arquivo binário de pesos da rede neural PyTorch.

```mermaid
graph TD
    classDef files fill:#f9f9f9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef models fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A[IMAGENS ORIGINAIS<br>data/raw/images] --> B(generate_masks.py)
    B -->|Classical HSV| C[MÁSCARAS ALVO<br>data/raw/masks]
    C --> D(split_dataset.py)
    
    D -->|80% Treino| E1[Conjunto Treino]
    D -->|20% Validação| E2[Conjunto Validação]
    
    E1 --> F(train_segmentation.py)
    E2 --> F
    F -->|Treina U-Net PyTorch| G[yolk_segmentation.pth]

    class A,C,E1,E2 files;
    class B,D,F scripts;
    class G models;
```

## Fluxo de Treinamento do Modelo DSM (Random Forest)
Este diagrama detalha como o total de imagens passam pela extração colorimétrica para treinar o estimador estatístico, incluindo a divisão exata observada nos seus relatórios de validação.

```mermaid
graph TD
    classDef files fill:#f9f9f9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef models fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A[100% IMAGENS + dsm_labels.csv] --> B(train_dsm.py)
    B -->|Segmentação + Extração| C[dsm_training_features.csv]
    
    C --> D1[80% Imagens<br>Treino]
    C --> D2[20% Imagens<br>Validação]
    
    D1 --> E[Random Forest Regressor]
    E --> F[dsm_random_forest.pkl]
    
    F --> G(Avaliação de Métricas)
    D2 --> G
    G --> H[Relatório Final<br>R² ~0.7653 / Margem ±1 ~80.23%]

    class A,C,D1,D2,H files;
    class B,G scripts;
    class F models;
```

## Uso do Sistema em Produção (Inference Pipeline Híbrido)
Este diagrama documenta a arquitetura de execução em tempo real do aplicativo Streamlit, detalhando o comportamento lógico do algoritmo de contingência (_fallback_).

```mermaid
graph TD
    classDef entry fill:#fffde7,stroke:#fbc02d,stroke-width:2px;
    classDef logic fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef action fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef out fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A[NOVA IMAGEM CARREGADA] --> B[Algoritmo Clássico HSV]
    
    B --> C{Máscara possui<br>≥ 100 pixels?}
    C -->|Sim| D[Usar Segmentação Clássica]
    C -->|Não / Falha| E[Usar IA: yolk_segmentation.pth]
    
    D --> F[MÁSCARA FINAL DA GEMA]
    E --> F
    
    F --> G[Extração de Cores<br>HSV + LAB]
    G --> H[Regressor: dsm_random_forest.pkl]
    H --> I[EXIBIÇÃO STREAMLIT<br>DSM Contínuo + Escala Comercial]

    class A entry;
    class C logic;
    class B,D,E,G,H action;
    class F,I out;
```
