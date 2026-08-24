# colovo

## Fluxo de Preparação e Treinamento da Segmentação (Auto-Labeling + U-Net)
Este diagrama detalha o processo desde as imagens brutas em formato de câmera até a exportação do arquivo binário de pesos da rede neural PyTorch.

```mermaid
graph TD
    %% Estilos e Cores Gerais
    classDef files fill:#f9f9f9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef models fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    
    A[IMAGENS ORIGINAIS<br>Formatos: .jpg, .jpeg, .png] -->|Caminho: data/raw/images| B(generate_masks.py)
    
    subgraph Pipeline Interno Clássico
        B --> B1[Leitura de Imagem via OpenCV]
        B1 --> B2[Conversão de Espaço: BGR para HSV]
        B2 --> B3[Aplicação de Limiares / Threshold]
        B3 --> B4[Operações Morfológicas de Limpeza]
    end
    
    B4 -->|Gera Máscaras Alvo| C[MÁSCARAS GERADAS<br>Formatos: .png binário]
    C -->|Caminho: data/raw/masks| D(split_dataset.py)
    
    subgraph Divisão Estatística do Dataset
        D --> D1[Garante Pareamento: Imagem <-> Mask]
        D1 --> D2[Separação Aleatória Estrita]
        D2 --> D3[Conjunto de Treino: Imagens + Máscaras]
        D2 --> D4[Conjunto de Validação: Imagens + Máscaras]
    end
    
    D3 -->|Alimenta o Treino| E(train_segmentation.py)
    D4 -->|Valida Época por Época| E
    
    subgraph Processo de Aprendizado de Máquina PyTorch
        E --> E1[Arquitetura U-Net ResNet Backboned]
        E1 --> E2[Entrada: Imagem RGB Tensor]
        E2 --> E3[Função de Perda: BCE + Dice Loss]
        E3 --> E4[Otimizador: Adam / SGD]
    end
    
    E4 -->|Salvamento dos Pesos Finais| F[yolk_segmentation.pth]
    
    class A,C,D3,D4 files;
    class B,D,E scripts;
    class F models;
```
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
Este diagrama detalha como as 426 imagens passam pela extração colorimétrica para treinar o estimador estatístico, incluindo a divisão exata observada nos seus relatórios de validação.

```mermaid
graph TD
    classDef files fill:#f9f9f9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef models fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A[426 IMAGENS ORIGINAIS + GABARITO DSM] -->|data/raw/images e dsm_labels.csv| B(train_dsm.py)
    
    subgraph Pipeline de Extração Colorimétrica de Features
        B --> B1[Segmenta Gema usando yolk_segmentation.pth]
        B1 --> B2[Filtro de Exclusão: Erosão de Borda Interna]
        B2 --> B3[Cálculo de Estatísticas da Região Válida]
        B3 --> B4[Extração das Medianas: Hue, Sat, Val, LAB_A, LAB_B]
    end
    
    B4 -->|Exportação de Dataset Tabular| C[dsm_training_features.csv]
    
    subgraph Divisão e Treino do Regressor Scikit-Learn
        C --> D1[Divisão Estatística dos Vetores]
        D1 -->|80% dos Dados| D2[Dataset de Treino<br>340 Gemas Extraídas]
        D1 -->|20% dos Dados| D3[Dataset de Validação<br>86 Gemas Extraídas]
        
        D2 --> E[Algoritmo Random Forest Regressor]
        E --> E1[Construção de N Árvores de Decisão]
    end
    
    E1 -->|Mapeamento de Ajustes de Pesos| F[dsm_random_forest.pkl]
    F -->|Executa Predição de Teste| G(Avaliação de Performance)
    D3 -->|Entrada Oculta para Validação| G
    
    subgraph Métricas de Qualidade Geradas
        G --> G1[Cálculo de R² Score ~0.7653]
        G1 --> G2[Cálculo de MAE / Erro Médio Absoluto]
        G2 --> G3[Aproximação Comercial Comercial: Tolerância ±1 DSM ~80.23%]
    end

    class A,C,D2,D3 files;
    class B,G scripts;
    class F models;
```
```mermaid
graph TD
    classDef files fill:#f9f9f9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    classDef scripts fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef models fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    A[426 IMAGENS + dsm_labels.csv] --> B(train_dsm.py)
    B -->|Segmentação + Extração| C[dsm_training_features.csv]
    
    C --> D1[340 Imagens<br>Treino]
    C --> D2[86 Imagens<br>Validação]
    
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

    A[NOVA IMAGEM CARREGADA] -->|Upload via interface Streamlit| B(Processamento Inicial)
    B -->|Leitura de Imagem e Normalização| C[Algoritmo Clássico HSV]
    
    subgraph Motor de Decisão Híbrido
        C --> D{A máscara possui<br>≥ 100 pixels válidos?}
        D -->|Sim| E[MÉTODO CLÁSSICO DEFINIDO]
        E -->|Usa Máscara do HSV Clássico| G[MÁSCARA FINAL DA GEMA]
        
        D -->|Não / Falha ou Gema Pequena| F[MÉTODO DE FALLBACK ACIONADO]
        F -->|Invoca Rede Neural U-Net PyTorch| H[yolk_segmentation.pth]
        H -->|Gera Máscara Baseada em IA| G
    end
    
    subgraph Extração de Features e Predição
        G --> I[Isolamento da Gema via bitwise_and]
        I --> J[Cálculo do Perfil Cromático da Região]
        J -->|Vetor: Hue, Sat, Val, Lab_A, Lab_B| K[Modelo Regressor DSM]
        K -->|Carrega dsm_random_forest.pkl| L[Predição Contínua do Valor]
    end
    
    subgraph Renderização de Interface Streamlit
        L --> M[DSM Estimado Exato Ex: 11.42]
        L --> N[Escala Comercial Arredondada Ex: 11]
        J --> O[Exibição de Métricas Individuais das Cores]
    end

    class A entry;
    class D logic;
    class E,F,H,K action;
    class M,N,O out;
```
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
