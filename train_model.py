import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)


class LetterVision:
    def __init__(self):
        self.model = self.build_model()

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

        x_train = x_train[..., None]
        x_test  = x_test[..., None]

        return (x_train, y_train), (x_test, y_test)

    def data_augmentation(self, rotation=0.1, zoom=0.1, translation=0.1, seed=42):
        return keras.Sequential([
            layers.RandomRotation(rotation, seed=seed),
            layers.RandomZoom(zoom, seed=seed),
            layers.RandomTranslation(translation, translation, seed=seed)
        ])

    def build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            self.data_augmentation(),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, x_train, y_train, epochs=5, batch_size=64):
        steps_per_epoch = len(x_train) // batch_size
        total_visto = steps_per_epoch * batch_size * epochs

        print(f"\nDataset original: {len(x_train)}")
        print(f"Com augmentation (efetivo): {total_visto}\n")

        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint("model.h5", save_best_only=True)
        ]

        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            shuffle=True
        )

    def set_seed(self, seed=42, tf="1"):
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = str(tf)
        os.environ["TF_CUDNN_DETERMINISTIC"] = str(tf)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def evaluate(self, x_test, y_test):
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
        idx = random.randint(0, len(x_test) - 1)

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