# Melatonina

Este repositorio contiene el código para la implementación de una cámara acústica. El objetivo es, a través de un array de micrófonos y una cámara convencional, aislar los distintos sonidos que producen los autos que circulan en una determinada vía e identificar los que tienen un sonido por encima del límite legal. También puede ser usado para identificar el uso excesivo de las bocinas en horarios de silencio.

Personalmente, me gustan los autos, tanto los de ciudad, como los clásicos, o los deportivos. Modificar los autos para lograr un mejor comportamiento es un hobby divertido, pero tiene su lugar y momento. Ese lugar y momento definitivamente no es en el medio de la ciudad a las dos de la mañana. Donde vivo, no hay ningún control de los ruidos de los autos durante su circulación, sólo en controles - en el mejor de los casos - anuales que parecen ser esquivados frecuentemente. La implementación de un sistema sencillo y de bajo costo podría permitir a las municipalidades garantizar una mejor calidad de vida.

## Implementación

El proyecto está implementado siguiendo los pasos de Pavlidi et al. (2013). En la carpeta `melatonin` (el código está implementado en inglés para facilitar el intercambio de ideas) se encuentra el paquete principal de detección.

## Referencias

Pavlidi, D., Griffin, A., Puigt, M., & Mouchtaris, A. (2013). Real-time multiple sound source localization and counting using a circular microphone array. IEEE Transactions on Audio, Speech, and Language Processing, 21(10), 2193-2206.