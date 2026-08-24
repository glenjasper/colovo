# colovo

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
