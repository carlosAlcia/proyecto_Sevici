# Sevici : Predicción del estado de estaciones Sevici


# Descripción del proyecto

¿Habrá bicicletas en esta estación mañana temprano? ¿Podré aparcar la bicicleta cerca del trabajo? Conocer la disponibilidad de bicicletas y aparcamientos del servicio Sevici puede ser de gran ayuda a la hora de decidir qué medio de transporte utilizar.

A partir de esta idea surge este proyecto: Poner a disposición de los usuarios un servicio de predicción del estado de las estaciones Sevici, usando modelos de machine learning, a partir de datos reales de utilización del servicio.

## Generación de la base de datos

Para poder desarrollar un modelo de machine learning capaz de predecir el estado de las estaciones Sevici, es necesario tener una base de datos que incluya esta información en el pasado. La API pública de JCDecaux permite obtener el estado de las estaciones en tiempo real, pero no ofrecen los datos históricos. Tras un intento sin éxito de comunicación con la empresa para tener acceso a estos datos, se opta por generar la base de datos usando los datos en tiempo real.

Así, se decidió crear un sistema automático de recopilación de los datos en tiempo real. Durante tres días y cada 10 minutos, se guardaría el estado actual de todas las estaciones en la base de datos. Los tres días seleccionados son los más representativos posible : 
- Jueves : Día laboral entre semana.
- Viernes : Día laboral, fin de semana.
- Sábado : Día no laboral.

Como es lógico, la limitación a 3 días impide utilizar el modelo de ML generado en la práctica. Para tener un buen modelo, haría falta recopilar datos durante muchos más días, para tener en cuenta factores como el clima, los festivos, las vacaciones... Sin embargo, para que el proyecto sea realizable en un plazo razonable, se ha decidido limitar la recopilación de datos reales a estos 3 días, y generar el resto de datos de forma artificial siguiendo argumentos lógicos.

### Obtención de los datos reales

Para obtener los datos reales cada 10 minutos durante tres días, se desarrolló un script Python (`create_db_from_api.py`) que interroga la API de JCDecaux e introduce los datos recibidos en una base de datos PostgreSQL. Para ejecutar periódicamente el script de Python, se creó una nueva entrada en la tabla **crontab**, que permite programar tareas en Linux (y MacOS). 

### Generación de datos sintéticos

Para generar los datos de 1 mes entero se han usado las siguientes suposiciones:
- Una temperatura inferior o superior a 10 y 25 grados, respectivamente, reduce el uso un 50%. El uso se mide como el número de stands disponibles para dejar las bicicletas: Si mucha gente utiliza el servicio, habrá pocas bicicletas disponibles y por lo tanto más espacios disponibles. Esto es una simplificación que tampoco tiene en cuenta que los trayectos pueden ser muchos pero muy cortos y que por lo tanto no impacte tanto en la disponibilidad de bicicletas.
- La lluvia reduce el uso un 80%.
- Un día festivo reduce el uso un 30%.
- Un día de viento fuerte reduce el uso un 50%.
- Un día de huelga en medios de transporte públicos aumenta el uso un 50%.

Estos factores se calculan para un día completo. No es realista ya que la temperatura, por ejemplo, varía según la hora del día, al igual que otros factores como la lluvia. En una aplicación real, estas medidas serían tomadas de fuentes externas y estarían ajustadas según el momento del día.
También se incluye un ruido aleatorio a los datos para dificultar el aprendizaje de los modelos.


## Entrenamiento de los modelos de ML
En la carpeta `prediction` se encuentra el archivo `training.py`. Este archivo permite elegir entre un modelo CatBoostRegressor o un modelo personalizado de red neuronal MLP. 

- CatBoostRegressor : Modelo basado en gradient boosting sobre árboles de decisión. No necesita apenas preprocesamiento de los datos. Captura fácilmente relaciones no lineales y combinaciones entre variables.
- MLP : Implementada en PyTorch. Arquitectura configurable con capas ocultas, dropout y normalización. Requiere normalizar los datos.

### Resultados obtenidos

Para evaluar los modelos, se separa el último día y otros puntos aleatorios de la base de datos antes del entrenamiento. Con estos puntos aleatorios se genera el set de test, con el que se puede estimar el rendimiento del modelo. Con los datos del último día se puede realizar una predicción de un día completo y realizar gráficas para visualizar mejor los resultados.

- Modelo CatBoostRegressor :
    Gráfica XY con set de test :
    !(Catboost.png)[./images/Catboost.png]
    En esta gráfica, cuánto más cerca estén los puntos del eje diagonal, mejor son las predicciones. Puede verse que el modelo es capaz de aprender y los resultados siguen claramente la tendencia de la diagonal.



# Estructura de archivos



### _NOTA IMPORTANTE_ 
No se trata de una aplicación en producción ni de un sistema completo funcional.
Su objetivo principal es servir como ejemplo técnico dentro de un portafolio personal, mostrando el uso de herramientas como Python, PostgreSQL, APIs públicas, visualización de datos, y conceptos de Machine Learning.

Si bien los datos utilizados son reales, el foco del proyecto está en la integración tecnológica, no en la explotación final de los resultados.