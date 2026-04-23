import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

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

    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),

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

    def train(self, x_train, y_train):
        self.model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            shuffle=True
        )

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
        accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        print("\n=== MÉTRICAS POR CLASSE ===")
        for i in range(10):
            print(
                f"Class {i}: "
                f"Precision={precision[i]:.4f} | "
                f"Recall={recall[i]:.4f} | "
                f"Specificity={specificity[i]:.4f} | "
                f"F1={f1_score[i]:.4f} | "
                f"Acc={accuracy_per_class[i]:.4f}"
            )

        macro_precision = tf.reduce_mean(precision)
        macro_recall = tf.reduce_mean(recall)
        macro_specificity = tf.reduce_mean(specificity)
        macro_f1 = tf.reduce_mean(f1_score)

        print("\n=== MÉTRICAS GERAIS ===")
        print(f"Loss: {test_loss:.4f}")
        print(f"Acurácia: {test_acc:.4f}")
        print(f"Precision (macro): {macro_precision:.4f}")
        print(f"Recall (macro): {macro_recall:.4f}")
        print(f"Specificity (macro): {macro_specificity:.4f}")
        print(f"F1-score (macro): {macro_f1:.4f}")

    def explain_with_lime(self, x_test, y_test, idx=None):
        if idx is None:
            idx = random.randint(0, len(x_test) - 1)

        image = x_test[idx]
        label = y_test[idx]

        image_input = image[None, ...]

        probs = self.model.predict(image_input, verbose=0)[0]

        def predict_fn(images):
            images = np.array(images).astype("float32")
            if images.shape[-1] == 3:
                images = np.mean(images, axis=-1, keepdims=True)

            if len(images.shape) == 2:
                images = images.reshape(1, 28, 28, 1)

            if len(images.shape) == 3:
                images = np.expand_dims(images, axis=-1)

            return self.model.predict(images, verbose=0)


        explainer = lime_image.LimeImageExplainer()

        explanation = explainer.explain_instance(
            image.squeeze(),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=500
        )

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=8,
            hide_rest=False
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(
            f"LIME Explanation\nReal: {label} | Pred: {np.argmax(probs)}",
            fontsize=12,
            fontweight='bold'
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print("\n=== PREDIÇÃO POR CLASSE ===")
        print("-" * 50)

        for i, p in enumerate(probs):
            print(f"Classe {i}: {p * 100:.2f}%")

        print("-" * 50)
        print(f"Classe prevista: {np.argmax(probs)}")

        print("\n=== RESUMO DA EXPLICAÇÃO LIME ===")
        print(f"Classe real: {label}")
        print(f"Classe prevista: {np.argmax(probs)}")
        print(f"Confiança: {np.max(probs) * 100:.2f}%")


def main():
    lv = LetterVision()
    (x_train, y_train), (x_test, y_test) = lv.load_data()

    lv.train(x_train, y_train)
    lv.evaluate(x_test, y_test)

    lv.model.save("model.h5")
    print("\nModelo salvo como model.h5")
    lv.explain_with_lime(x_test, y_test)

if __name__ == "__main__":
    main()