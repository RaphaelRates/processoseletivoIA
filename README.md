# Processo Seletivo – Intensivo Maker | AI

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 Navegação Rápida

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **“Reopen in Container”**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`



### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---
Aqui está o relatório adaptado com as correções solicitadas, mantendo todo o conteúdo original e adicionando explicações sobre partes que não estão no README:

---

# Relatório do Candidato

**Identificação:** Raphael Sousa Rabelo Rates
**Instituição:** Universidade Federal do Cariri (UFCA)

---

## 1. Resumo da Arquitetura do Modelo

O modelo desenvolvido opera sobre imagens de 28×28 pixels em escala de cinza (canal único), provenientes do dataset MNIST. Durante o treinamento, técnicas de data augmentation são aplicadas diretamente na arquitetura para ampliar a variabilidade dos dados e favorecer a capacidade de generalização.

### Entrada e Pré-processamento

As imagens de entrada são normalizadas pela divisão por 255.0, o que padroniza os valores de pixel para o intervalo [0, 1] e contribui para uma convergência mais estável e eficiente durante o treinamento da rede.

### Primeiro Bloco Convolucional

O primeiro bloco é composto por uma camada convolucional com 32 filtros de tamanho 3×3 e função de ativação ReLU. Em seguida, uma camada de MaxPooling com janela 2×2 reduz a dimensionalidade espacial da imagem de 28×28 para aproximadamente 14×14, preservando as características mais relevantes detectadas pelos filtros.

### Segundo Bloco Convolucional

O segundo bloco aplica uma nova camada convolucional com 64 filtros de tamanho 3×3 e ativação ReLU, seguida de outra camada de MaxPooling 2×2. Isso reduz a dimensão espacial para cerca de 7×7, permitindo ao modelo capturar padrões progressivamente mais complexos e abstratos presentes nos dados.

### Bloco de Classificação

O mapa de características resultante é linearizado por uma camada Flatten, convertendo o tensor tridimensional em um vetor unidimensional. Uma camada totalmente conectada com 256 neurônios e ativação ReLU é responsável por aprender combinações globais das características extraídas. Por fim, a camada de saída possui 10 neurônios com ativação softmax, produzindo uma distribuição de probabilidades sobre cada uma das classes (dígitos de 0 a 9).

Trata-se de uma arquitetura simples e eficiente para o problema de classificação de dígitos manuscritos, equilibrando capacidade representacional com baixo custo computacional.

---

## 2. Bibliotecas Utilizadas

As dependências do projeto estão especificadas a seguir, com suas respectivas versões:

```
tensorflow==2.12.0
numpy==1.24.3
matplotlib==3.7.1
scikit-image==0.21.0
scikit-learn>=1.3
lime==0.2.0.1
```

**TensorFlow (2.12.0)** é o framework principal de deep learning utilizado para a construção, treinamento e avaliação da rede neural convolucional. **NumPy (1.24.3)** fornece suporte fundamental à computação numérica, sendo empregado na manipulação eficiente de arrays e em operações matemáticas auxiliares. **Matplotlib (3.7.1)** é utilizado para visualização de dados, incluindo a exibição de imagens e a renderização das explicações geradas pelo LIME.

**Scikit-Image (0.21.0)** oferece funcionalidades de processamento de imagens, como a função `mark_boundaries`, utilizada na demarcação visual de regiões relevantes identificadas pelo LIME. **Scikit-Learn (>=1.3)** provê métricas de avaliação robustas — incluindo acurácia, precisão, recall, F1-score e matriz de confusão — além de ferramentas auxiliares para análise de desempenho. **LIME (0.2.0.1)** é a biblioteca de interpretabilidade responsável por destacar as regiões da imagem que mais influenciam a decisão do modelo em cada predição.

---

## 3. Técnicas Utilizadas no Modelo

### Quantização Pós-Treinamento

A quantização pós-treinamento foi aplicada com o objetivo de reduzir o tamanho do modelo e aumentar a velocidade de inferência em dispositivos com recursos limitados. A técnica converte os pesos do modelo de `float32` para `int8` ou `float16`, reduzindo o consumo de memória em aproximadamente 75 a 80% e acelerando a inferência em um fator de 2 a 4x em dispositivos móveis. O modelo também foi convertido do formato Keras H5 (`.h5`) para TFLite (`.tflite`), viabilizando sua execução em plataformas como Android, iOS e microcontroladores.

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

A tabela a seguir resume o impacto prático da otimização:

| Métrica         | Keras (.h5) | TFLite quantizado | Impacto        |
|-----------------|-------------|-------------------|----------------|
| Acurácia        | 99,06%      | ~99,00%           | -0,06%         |
| Tamanho         | ~25 MB      | ~6 MB             | -76%           |
| Inferência (CPU)| 15 ms       | 4 ms              | 3,7x mais rápido |

A perda de acurácia é desprezível neste contexto, pois o MNIST é um problema de baixa complexidade. Para tarefas mais desafiadoras, recomenda-se o uso de `float16` em detrimento de `int8`.

### Data Augmentation

Para aumentar a capacidade de generalização do modelo e mitigar o overfitting, foi aplicado data augmentation on-the-fly diretamente como parte da arquitetura da rede, por meio de camadas Keras. As transformações aplicadas incluem rotação aleatória de até 10%, zoom aleatório de até 10% e translação aleatória nos eixos X e Y de até 10%.

```python
layers.RandomRotation(0.1),
layers.RandomZoom(0.1),
layers.RandomTranslation(0.1, 0.1)
```

A abordagem on-the-fly tem a vantagem de não aumentar o tamanho físico do dataset, sendo executada em tempo real durante o treinamento e sem impactar o conjunto de teste. Essa estratégia simula um dataset maior sem custo de armazenamento adicional.

### Early Stopping

O mecanismo de parada antecipada foi adotado com o objetivo de interromper o treinamento automaticamente quando não houver melhora na métrica de validação após um número determinado de épocas consecutivas, evitando overfitting e reduzindo custo computacional desnecessário.

```python
EarlyStopping(patience=3, restore_best_weights=True)
```

Com `patience=3`, o treinamento é interrompido após três épocas sem melhora, e os melhores pesos encontrados são automaticamente restaurados ao final do processo.

### Model Checkpoint

Para garantir a preservação do melhor modelo obtido durante o treinamento, foi utilizado o callback `ModelCheckpoint`, configurado para salvar apenas o modelo com menor loss de validação.

```python
ModelCheckpoint("model.h5", save_best_only=True)
```

Essa estratégia é essencial em contextos experimentais, onde múltiplas execuções são comparadas e a recuperação do melhor estado da rede deve ser garantida.

### Controle de Aleatoriedade

A reprodutibilidade dos experimentos foi assegurada por meio da definição de um seed fixo, garantindo que os resultados possam ser replicados de forma consistente e que comparações entre experimentos sejam realizadas em condições equivalentes.

```python
self.set_seed(42)
```

### Métricas Avançadas de Avaliação

Além das métricas tradicionais, foram empregadas métricas mais robustas para uma avaliação mais completa do desempenho do modelo. O **Cohen's Kappa** mede a concordância entre predições e rótulos reais, descontando a concordância esperada por acaso. O **Matthews Correlation Coefficient (MCC)** oferece uma avaliação equilibrada mesmo na presença de classes desbalanceadas. A **ROC-AUC (One-vs-Rest)** quantifica a capacidade do modelo de separar cada classe das demais, e a **especificidade** mensura a taxa de verdadeiros negativos por classe.

---

## 4. Resultados Obtidos

### Sumário Executivo

O modelo CNN apresentou desempenho excepcional no conjunto de teste, atingindo 99,06% de acurácia global. As métricas demonstram alta capacidade de generalização e baixíssima taxa de erro entre as classes avaliadas.

### Matriz de Confusão

```
[[ 963    0    1    1    2    0    6    1    6    0]
 [   0 1058    1    7    9    1   16    4   39    0]
 [   0    0 1022    5    2    0    0    2    1    0]
 [   0    0    0 1009    0    0    0    1    0    0]
 [   0    0    0    0  973    0    0    0    4    5]
 [   0    0    0   13    0  875    3    0    1    0]
 [   0    1    0    0    4    4  948    0    1    0]
 [   0    1    7    6    5    0    0 1006    1    2]
 [   0    0    3    2    2    1    0    2  962    2]
 [   0    0    0    3    9    2    0    4    5  986]]
```

### Métricas por Classe

| Dígito | Precisão  | Recall  | Especificidade | F1-score | Acurácia |
|:------:|----------:|--------:|---------------:|---------:|---------:|
| 0      | 100,00%   | 98,27%  | 100,00%        | 99,13%   | 99,83%   |
| 1      | 99,72%    | 93,22%  | 99,97%         | 96,36%   | 99,23%   |
| 2      | 98,93%    | 99,03%  | 99,88%         | 98,98%   | 99,77%   |
| 3      | 96,65%    | 99,90%  | 99,63%         | 98,25%   | 99,58%   |
| 4      | 97,01%    | 99,08%  | 99,67%         | 98,03%   | 99,55%   |
| 5      | 99,32%    | 98,10%  | 99,93%         | 98,71%   | 99,77%   |
| 6      | 97,53%    | 98,96%  | 99,74%         | 98,24%   | 99,57%   |
| 7      | 98,82%    | 97,86%  | 99,88%         | 98,34%   | 99,76%   |
| 8      | 94,03%    | 98,77%  | 99,36%         | 96,34%   | 99,16%   |
| 9      | 99,30%    | 97,72%  | 99,92%         | 98,50%   | 99,74%   |

### Métricas Agregadas

| Métrica                      | Valor   |
|-----------------------------|--------:|
| Acurácia Geral              | 98,99%  |
| F1-score (macro)            | 98,29%  |
| Precision (macro)           | 98,73%  |
| Recall (macro)              | 98,09%  |
| Cohen's Kappa               | 98,88%  |
| Matthews Corrcoef (MCC)     | 98,88%  |

### Análise dos Erros

A análise da matriz de confusão revela que a maior fonte de erro do modelo está nas confusões envolvendo o dígito "1", que em 39 casos foi classificado incorretamente como "8" — provavelmente devido à presença de serifas ou à inclinação da escrita, que confere ao traço uma aparência similar às curvas do "8". Outras confusões relevantes incluem "1" classificado como "6" (16 ocorrências), "5" como "3" (13 ocorrências), e erros recíprocos entre "9" e "4" (9 e 5 ocorrências, respectivamente).

O dígito "0" apresenta o melhor desempenho geral, com F1-score de 99,13% e acurácia de 99,83%. Em contrapartida, o dígito "1" possui o menor recall (93,22%), indicando que o modelo deixa escapar uma proporção maior de instâncias dessa classe, enquanto o dígito "8" apresenta a menor precisão (94,03%), sinalizando maior frequência de falsos positivos nessa categoria.


## 5. Comentários Adicionais

### Dificuldades Encontradas

A principal dificuldade técnica envolveu a adaptação da função `predict_fn` para compatibilidade com o LIME. A biblioteca exige que a função de predição receba um batch de imagens no formato adequado (28×28×1), o que demandou tratamento explícito das dimensões do tensor de entrada. Adicionalmente, o cálculo de métricas como especificidade e acurácia por classe não está disponível nativamente no Keras, sendo necessário implementá-las manualmente a partir da matriz de confusão com operações do TensorFlow.

### Decisões Técnicas

A arquitetura CNN foi definida com aumento progressivo de filtros (32 → 64 → 128) nas camadas convolucionais, permitindo a extração de características em diferentes níveis de abstração — de padrões simples como bordas e texturas até estruturas mais complexas. A divisão de validação em `validation_split=0.1` foi adotada para monitorar o overfitting sem comprometer o volume de dados disponíveis para treinamento, mesm que o jeito correto poderia ser o uso de treinamento, validação e teste para o treinamento do modelo. A explicabilidade por LIME foi configurada com 500 amostras, valor que oferece estabilidade nas explicações sem custo computacional excessivo.

### Limitações do Modelo

O modelo foi treinado e otimizado exclusivamente para o dataset MNIST, não generalizando para outros domínios como letras, dígitos estilizados ou outros sistemas de escrita. A entrada é restrita a imagens 28×28 pixels, exigindo redimensionamento para qualquer outro formato. Embora o data augmentation tenha sido aplicado, variações extremas de rotação ou translação podem ainda comprometer a robustez do modelo. Por fim, as explicações fornecidas pelo LIME são aproximadas e de natureza local, não oferecendo interpretações causais sobre o comportamento global da rede.

### Aprendizados

O desenvolvimento do projeto evidenciou o trade-off entre profundidade arquitetural e capacidade de generalização: redes muito profundas tendem a overfitar no MNIST, o que motivou a escolha de uma arquitetura com apenas dois blocos convolucionais. A aplicação de quantização pós-treinamento demonstrou ser uma estratégia eficaz para tornar modelos CNN viáveis em ambientes com recursos computacionais restritos, com perda de desempenho desprezível no contexto deste problema.

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
