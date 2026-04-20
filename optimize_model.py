import tensorflow as tf
import os


class ModelOptimizer:
    def __init__(self, model_path="model.h5", tflite_path="model.tflite"):
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        print("Modelo carregado com sucesso")

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()

    def save_tflite_model(self, tflite_model):
        with open(self.tflite_path, "wb") as f:
            f.write(tflite_model)

    def print_size(self):
        size = os.path.getsize(self.tflite_path)
        print(f"Tamanho do modelo TFLite: {size / 1024:.2f} KB")

    def optimize(self):
        self.load_model()

        tflite_model = self.convert_to_tflite()

        self.save_tflite_model(tflite_model)

        self.print_size()

        print("\nModelo otimizado salvo como model.tflite")


def main():
    optimizer = ModelOptimizer()
    optimizer.optimize()


if __name__ == "__main__":
    main()