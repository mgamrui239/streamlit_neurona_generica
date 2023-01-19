import streamlit as st
import numpy as np
from PIL import Image

image = Image.open('neurona.jpg')

st.image(image)

class Neuron:
  def __init__(self, weights = 0, bias = 0, func = ""):
    self.weights = weights
    self.bias = bias
    self.func = func

  def run(self, input_data):
    x = np.dot(input_data, self.weights) + self.bias
    if self.func == "Sigmoide":
      return 1/(1 + np.exp(-x))
    if self.func == "ReLU":
      if self.bias > 0:
        return x
      else:
        return 0
    if self.func == "Tangente Hiperbólica":
      return np.tanh(x)

st.title("Simulación Neurona")

st.header("Neurona completa")

main = st.slider("Numero de datos", 1, 10)

columnas = st.columns(main)
w = np.empty(main)
x = np.empty(main)

for i in range(main):
    with columnas[i]:
        w[i] = st.slider(f'w{i}', 0.0, 10.0)
        x[i] = st.number_input(f'x{i}')

b = st.number_input("Bias")

act_function = st.selectbox(
    'Selecciona una función de activación:',
    ('Sigmoide','ReLU', 'Tangente Hiperbólica')
)

if st.button("Calcular la salida", key='buttontab3'):
    n1 = Neuron(weights=w, bias=b, func=act_function)
    output = n1.run(input_data=x)
    st.text(output)