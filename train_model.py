import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import optuna


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

    def data_augmentation(self, rotation=0.1, zoom=0.1, translation=0.1):
        return keras.Sequential([
            layers.RandomRotation(rotation),
            layers.RandomZoom(zoom),
            layers.RandomTranslation(translation, translation)
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

    def build_model_with_params(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        rotation = trial.suggest_float("rotation", 0.0, 0.2)
        zoom = trial.suggest_float("zoom", 0.0, 0.2)
        translation = trial.suggest_float("translation", 0.0, 0.2)

        filters1 = trial.suggest_categorical("filters1", [32, 64])
        filters2 = trial.suggest_categorical("filters2", [64, 128])
        dense_units = trial.suggest_categorical("dense", [64, 128, 256])

        data_aug = self.data_augmentation(rotation, zoom, translation)

        model = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            data_aug,

            layers.Conv2D(filters1, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(filters2, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
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

    def objective(self, trial, x_train, y_train):
        model = self.build_model_with_params(trial)

        history = model.fit(
            x_train, y_train,
            epochs=3, 
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )

        return max(history.history['val_accuracy'])

    def set_seed(self, seed=42):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)

        y_pred_probs = self.model.predict(x_test, verbose=0)
        y_pred = tf.argmax(y_pred_probs, axis=1)

        cm = tf.math.confusion_matrix(y_test, y_pred)
        print("\n=== MATRIZ DE CONFUSÃO ===")
        print(cm.numpy())

        cm = tf.cast(cm, tf.float32)

        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp
        tn = tf.reduce_sum(cm) - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        f1_score  = 2 * precision * recall / (precision + recall + 1e-7)

        print("\n=== MÉTRICAS GERAIS ===")

        print(f"Acurácia            : \033[92m{test_acc:.4f}\033[0m")
        print(f"Precision (macro)   : \033[96m{tf.reduce_mean(precision):.4f}\033[0m")
        print(f"Recall (macro)      : \033[93m{tf.reduce_mean(recall):.4f}\033[0m")
        print(f"Specificity (macro) : \033[95m{tf.reduce_mean(specificity):.4f}\033[0m")
        print(f"F1-score (macro)    : \033[91m{tf.reduce_mean(f1_score):.4f}\033[0m")


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

    print("\n=== OTIMIZAÇÃO COM OPTUNA ===")

    study = optuna.create_study(
        study_name="mnist_fast",
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1,
            n_warmup_steps=2,
            interval_steps=1
        ),
    )

    study.optimize(
        lambda trial: lv.objective(trial, x_train, y_train),
        n_trials=10
    )

    print("Melhores parâmetros:", study.best_params)

    lv.model = lv.build_model_with_params(study.best_trial)

    lv.train(x_train, y_train)
    lv.evaluate(x_test, y_test)

    lv.model.save("model.h5")
    print("\nModelo salvo como model.h5")

    lv.explain_with_lime(x_test, y_test)


if __name__ == "__main__":
    main()