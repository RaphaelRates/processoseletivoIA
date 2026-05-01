# =========================
# Bibliotecas padrão
# =========================

# Manipulação de arquivos, diretórios e variáveis de ambiente
import os

# Controle de aleatoriedade (reprodutibilidade)
import random


# =========================
# Bibliotecas numéricas
# =========================

# Operações com arrays e álgebra numérica
import numpy as np


# =========================
# Deep Learning (TensorFlow / Keras)
# =========================

# Framework principal para construção e treinamento de modelos
import tensorflow as tf

# API de alto nível para definição de redes neurais
from tensorflow import keras
from tensorflow.keras import layers

# Callbacks para controle do treinamento
# - EarlyStopping: interrompe treino quando não há melhoria
# - ModelCheckpoint: salva o melhor modelo durante o treino
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================
# Explicabilidade de modelos
# =========================

# LIME para explicação de predições em imagens
from lime import lime_image

# Função para destacar regiões importantes na imagem
from skimage.segmentation import mark_boundaries


# =========================
# Visualização
# =========================

# Plotagem de gráficos e imagens
import matplotlib.pyplot as plt


# =========================
# Métricas de avaliação
# =========================

from sklearn.metrics import (
    accuracy_score,        # Acurácia geral
    precision_score,       # Precisão (positivos corretos)
    recall_score,          # Sensibilidade (recall)
    f1_score,              # Média harmônica entre precisão e recall
    confusion_matrix,      # Matriz de confusão
    classification_report, # Relatório completo de métricas
    roc_auc_score,         # Área sob a curva ROC
    cohen_kappa_score,     # Concordância além do acaso
    matthews_corrcoef      # Correlação de Matthews (robusta p/ desbalanceamento)
)

EPOCHS = 5
BATCH_SIZE = 64
SEED = 42
PATH_MODEL = "model.h5"
PATIENCE = 3
VALIDATION_SPLIT = 0.1

class LetterVision:
    def __init__(self):
        """
            Classe responsável por todo o pipeline de um modelo de visão computacional
            aplicado ao dataset MNIST, incluindo:
            - carregamento e pré-processamento dos dados
            - definição da arquitetura da rede neural
            - treinamento com callbacks
            - avaliação com múltiplas métricas
            - explicabilidade via LIME
        """
        self.model = self.build_model()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    def load_data(self):
        """
        Carrega o dataset MNIST e realiza o pré-processamento básico:
        - normalização dos pixels para o intervalo [0,1]
        - expansão do canal para formato (H, W, 1)
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

        x_train = x_train[..., None]
        x_test  = x_test[..., None]

        return (x_train, y_train), (x_test, y_test)

    def data_augmentation(self, rotation=0.1, zoom=0.1, translation=0.1, seed=SEED):
        """
        Define um pipeline de data augmentation para aumentar a variabilidade dos dados:
        - rotações leves
        - zoom aleatório
        - translações horizontais e verticais

        Isso ajuda a reduzir overfitting e melhorar a generalização.
        """
        return keras.Sequential([
            layers.RandomRotation(rotation, seed=seed),
            layers.RandomZoom(zoom, seed=seed),
            layers.RandomTranslation(translation, translation, seed=seed)
        ])

    def build_model(self):
        """
            Constrói e compila uma CNN simples para classificação de dígitos (MNIST).

            Arquitetura:
            - Entrada: imagens 28x28 em escala de cinza (1 canal).
            - Data augmentation: aplica transformações leves para melhorar generalização.

            Bloco convolucional:
            - Conv2D(32, 3x3, ReLU):
            extrai padrões básicos (bordas, traços).
            - MaxPooling(2x2):
            reduz dimensionalidade espacial.
            - Conv2D(64, 3x3, ReLU):
            captura padrões mais complexos.
            - MaxPooling(2x2):
            reduz novamente a resolução.

            Classificação:
            - Flatten:
            transforma o mapa de features em vetor 1D.
            - Dense(128, ReLU):
            aprende combinações globais das features.
            - Dense(10, Softmax):
            gera probabilidades para as classes (0–9).

            Compilação:
            - Adam: otimizador adaptativo padrão.
            - Sparse Categorical Crossentropy:
            adequado para rótulos inteiros.
            - Accuracy: métrica de desempenho.

            Obs:
            - Modelo leve e eficiente para MNIST.
            - Utiliza apenas 2 camadas convolucionais (dentro do limite de até 3).
            """
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            self.data_augmentation(),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, validation_split=VALIDATION_SPLIT, seed=SEED):
        """
            Treina o modelo com os dados fornecidos.

            - Calcula e exibe o total de amostras vistas considerando épocas e batch size.
            - Usa EarlyStopping para evitar overfitting e restaurar os melhores pesos.
            - Usa ModelCheckpoint para salvar o melhor modelo durante o treino.
            - Treina com validação (10%) e embaralhamento dos dados.
        """
        steps_per_epoch = len(x_train) // batch_size
        total_visto = steps_per_epoch * batch_size * epochs

        print(f"\nDataset original: {len(x_train)}")
        print(f"Com augmentation (efetivo): {total_visto}\n")

        callbacks = [
            EarlyStopping(patience=PATIENCE, restore_best_weights=True),
            ModelCheckpoint(PATH_MODEL, save_best_only=True)
        ]
            
        print("\n=== CONFIGURAÇÃO DE TREINO ===")
        print(f"Dataset original : {len(x_train)}")
        print(f"Total visto      : {total_visto}")
        print(f"Epochs           : {epochs}")
        print(f"Batch size       : {batch_size}")
        print("Callbacks         : EarlyStopping + ModelCheckpoint")
        print(f"Patience         : {patience}")
        print(f"Validation split : {validation_split}")
        print(f"Seed             : {seed}")
        print("="*35 + "\n")

        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            shuffle=True
        )

    def set_seed(self, seed=42, deterministic=True):
        """
            Define a semente global para reprodutibilidade do experimento.

            - Configura variáveis de ambiente do Python e TensorFlow para execução determinística.
            - Controla fontes de aleatoriedade: random, NumPy e TensorFlow.
            - Garante que resultados sejam reproduzíveis entre execuções (quando possível).
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1" if deterministic else "0"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1" if deterministic else "0"

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def evaluate(self, x_test, y_test):
        """
            Avalia o modelo no conjunto de teste.

            - Gera predições e probabilidades por classe.
            - Calcula e exibe a matriz de confusão.
            - Deriva a especificidade por classe e apresenta a média.
            - Computa métricas principais:
            accuracy, precision, recall, F1 (macro),
            Cohen’s Kappa e Matthews Correlation Coefficient (MCC).
            - Calcula ROC-AUC (OvR) quando possível.
            - Exibe relatório completo de classificação por classe.
        """
        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print("\n=== MATRIZ DE CONFUSÃO ===")
        cm = confusion_matrix(y_test, y_pred)
        tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
        fp = cm.sum(axis=0) - np.diag(cm)

        specificity = tn / (tn + fp + 1e-7)
        print(cm)

        print("\n=== MÉTRICAS GERAIS ===")

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        try:
            roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        except:
            roc_auc = None

        print(f"Acurácia            : \033[92m{acc:.4f}\033[0m")
        print(f"Precision (macro)   : \033[96m{precision:.4f}\033[0m")
        print(f"Recall (macro)      : \033[93m{recall:.4f}\033[0m")
        print(f"F1-score (macro)    : \033[91m{f1:.4f}\033[0m")
        print(f"Cohen Kappa         : \033[95m{kappa:.4f}\033[0m")
        print(f"Matthews Corrcoef   : \033[94m{mcc:.4f}\033[0m")
        print(f"Specificity (macro) : \033[95m{np.mean(specificity):.4f}\033[0m")

        if roc_auc is not None:
            print(f"ROC-AUC (OvR)       : \033[92m{roc_auc:.4f}\033[0m")

        print("\n=== RELATÓRIO COMPLETO ===")
        print(classification_report(y_test, y_pred))


    def explain_with_lime(self, x_test, y_test):
        """
            Gera explicação local de uma predição usando LIME.

            - Seleciona uma amostra aleatória do conjunto de teste.
            - Obtém probabilidades do modelo e classe predita.
            - Define função de predição compatível com o LIME
            (ajusta formato e converte RGB → grayscale se necessário).
            - Executa o LIME para identificar regiões mais relevantes da imagem.
            - Exibe visualização com superpixels destacados.
            - Imprime um dashboard com probabilidades por classe,
            destacando classe real, predita e nível de confiança.
        """
        idx = random.randrange(len(x_test))

        image = x_test[idx]
        label = y_test[idx]

        probs = self.model.predict(image[None, ...], verbose=0)[0]
        pred_class = np.argmax(probs)

        def predict_fn(images):
            images = np.array(images).astype("float32")

            if images.shape[-1] == 3:
                images = np.mean(images, axis=-1, keepdims=True)

            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=-1)

            return self.model.predict(images, verbose=0)

        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image.squeeze(),
            predict_fn,
            top_labels=1,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=8
        )

        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"Real: {label} | Pred: {np.argmax(probs)}")
        plt.axis('off')
        plt.show()
        print("\n" + "="*50)
        print("DASHBOARD DE PREDIÇÃO")
        print("="*50)

        for i, p in enumerate(probs):
            if i == pred_class and i == label:
                marker = "\033[92m(CORRETO)\033[0m"
            elif i == pred_class:
                marker = "\033[93m(PREDITO)\033[0m"
            elif i == label:
                marker = "\033[94m(REAL)\033[0m"
            else:
                marker = ""

            print(f"Classe {i:>2}: {p*100:6.2f}% {marker}")

        print("-"*50)
        print(f"Classe real     : \033[94m{label}\033[0m")
        print(f"Classe predita  : \033[93m{pred_class}\033[0m")
        print(f"Confiança       : {np.max(probs)*100:.2f}%")

        if pred_class == label:
            print(f"Resultado       : \033[92mACERTO\033[0m")
        else:
            print(f"Resultado       : \033[91mERRO\033[0m")

        print("="*50 + "\n")

def main():
    lv = LetterVision()
    lv.set_seed(42)
    (x_train, y_train), (x_test, y_test) = lv.load_data()

    lv.model = lv.build_model()

    lv.train(x_train, y_train)
    lv.evaluate(x_test, y_test)

    lv.model.save("model.h5")
    print("\nModelo salvo como model.h5")

    lv.explain_with_lime(x_test, y_test)


if __name__ == "__main__":
    main()