import os
import numpy as np
import tensorflow as tf


class ModelOptimizer:
    def __init__(self, model_path="model.h5", tflite_path="model.tflite"):
        self.model_path = model_path
        self.tflite_path = tflite_path
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Modelo carregado: {self.model_path}")
        self.model.summary()

    def convert_to_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        print("Conversão para TFLite concluída (Dynamic Range Quantization aplicada)")
        return tflite_model

    def save_tflite_model(self, tflite_model):
        with open(self.tflite_path, "wb") as f:
            f.write(tflite_model)

    def print_size_comparison(self):
        original_size = os.path.getsize(self.model_path) / 1024
        tflite_size = os.path.getsize(self.tflite_path) / 1024
        reduction = (1 - tflite_size / original_size) * 100

        print(f"\nTamanho original (.h5):    {original_size:.2f} KB")
        print(f"Tamanho otimizado (.tflite): {tflite_size:.2f} KB")
        print(f"Reducao de tamanho:          {reduction:.1f}%")

    def validate_tflite(self):
        """Executa uma inferência de teste para confirmar que o modelo TFLite está funcional."""
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dummy_input = np.zeros(input_details[0]["shape"], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        print(f"\nValidacao TFLite: inferencia de teste bem-sucedida (output shape: {output.shape})")

    def optimize(self):
        self.load_model()

        tflite_model = self.convert_to_tflite()
        self.save_tflite_model(tflite_model)

        self.print_size_comparison()
        self.validate_tflite()

        print(f"\nModelo otimizado salvo em: {self.tflite_path}")


def main():
    optimizer = ModelOptimizer()
    optimizer.optimize()


if __name__ == "__main__":
    main()