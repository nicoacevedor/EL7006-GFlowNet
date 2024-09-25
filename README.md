# Causality-GFN 

## 1. Prepare environment

```Shell
conda create -n causality-GFN python=3.10 -y
conda activate causality-GFN
pip install --upgrade pip
pip install -e .
```

## 2. Running the Code

Para ejecutar el código, puedes utilizar los cuadernos de Jupyter proporcionados en el directorio `notebooks`. Por ejemplo, puedes comenzar con el [tutorial de GFlowNet](notebooks/GFlowNet_tutorial.ipynb) para comprender los conceptos básicos de GFlowNets y cómo se implementan en este proyecto.

## Causalidad y GFlowNets

### Descubriendo la causa de las actividades neuronales


Modelar y simular una red de neuronas con relaciones causales entre ellas. Tenemos una matriz de conectividad causal A que representa cómo las neuronas se influyen mutuamente. Existe una dependencia temporal (causalidad) el estado de la red en el tiempo t+1 depende de su estado en el tiempo t. Podemos utilizar GFlowNets para aprender la estructura representada por A explorando el espacio de posibles matrices de conectividad. Utilizando los datos de series temporales simuladas como entrada, podemos diseñar una GFN que genere matrices de conectividad candidatas entrenándola para maximizar la probabilidad de los datos observados dadas las estructuras generadas. A continuación, podemos utilizar la GFlowNet entrenada para muestrear y clasificar las estructuras causales probables. Si utilizamos la GFlowNet entrenada para muestrear y clasificar las estructuras causales probables, obtendremos una distribución sobre las posibles estructuras en lugar de una estimación puntual.

Finalmente estudiar lo siguiente:
- Simplificar el problema a matrices de conectividad binarias {0,1} de tamaño fijo y estudiar cómo se comporta la GFlowNet.
- Factibilidad de utilizar GFlowNets para inferir la estructura causal de una red de neuronas con estados en el continuo.
- ¿Podemos construir un modelo que tenga mejor desempeño que el estimador más simple basado en correlación para un número elevado de neuronas?

### Descubriendo la dínamica de un péndulo en un carro

Se probee la implementación del péndulo como un sistema más complejo donde la idea es encontrar la matriz del sistema discretizado o una solución a la ecuación de forma implícita. Para obtener datos del péndulo, puedes utilizar las función [`pendulum_cart_ode`](pendulum/pendulum_simulation.py) y emular el comportamiento del péndulo visualmente con la funciones [`draw_pendulum`](pendulum/pendulum_simulation.py), [`draw_pendulum_frame`](pendulum/pendulum_simulation.py).

De esta etapa se espera:
- Implementar un modelo de GFlowNet que pueda aprender la matriz del sistema discretizado del péndulo o una solución a la ecuación de forma implícita. Si se requiere cambiar el sistema por otro más simple hablar con el mentor.
- Estudiar cómo se comporta la GFlowNet en este sistema más complejo entendiendo sus limitaciones (dimensionalidad del problema, cantidad de datos, no linealidades, convergencia, etc).

### Descubriendo el conectoma causal del cerebro (Opcional)
El conectoma se refiere a la conectividad funcional que crea el cerebro cuando un sujeto está en reposo. Este se construye a partir de señales temporales (BOLD, respuesta hemodinámica) y permite definir en cierta forma la relación estadística entre par de regiones del cerebro. 
Generalmente para descubrir las relaciones entre ciertas regiones del cerebro se utiliza la correlación. Sin embargo la correlación tiene una limitante asociada a la cantidad de variables que interactúan en un sistema para encontrar la causa del fenómeno. Por lo tanto en este problema se plantea descubrir la relación causal de estas señales temporales usando GFN. 

Finalmente estudiar lo siguiente:

- Simplificar el problema a matrices de conectividad binarias {0,1} de tamaño fijo y estudiar cómo se comporta la GFlowNet.
- Abordar el problema a matrices de conectividad de tamaño fijo en un espacio continuo. 



### Datasets

- **Medium**: Simulated Network Population [x]
- **Hard**: Simulated Physical Problem [x]
- **Extremely Hard**: Real Resting State Images (13 Controles + 3 Pacientes) [-] https://drive.google.com/drive/folders/11nfumF8GiOD_dFqHYFosG95ifJ-F_iko?usp=sharing

