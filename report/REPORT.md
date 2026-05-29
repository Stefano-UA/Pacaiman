# Reporte

[GitHub Repo](https://github.com/Stefano-UA/Pacaiman)

## Introducción

## Parte 1 - Nuevas Heuristicas
El código de ``pacman`` contaba con una función de evaluación del `NeuralAgent`, la cual contaba con dos factores heurísitocos básicos:
- Factor 1: recompensa inversamente proporcional a la distancia al coco más cercano
- Factor 2: penalización fija de -200 si añgún fantasma se encuentra a distancia de <= 2

Para esta tarea, hemos añadido dos nuevos factores heurísticos adicionales a partir de los dos factores iniciales:

#### Heurística 3 - Huida 
```python
for ghost_state in ghost_states:
    if ghost_state.scaredTimer == 0:
        ghost_pos = ghost_state.getPosition()
        ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
        score -= 150.0 / (ghost_distance + 0.5)
```
Esta heurísitca aplica una penalización continua e inversamente proporcional a la distancia para cada fanstasma que no esté asustado (independientemente de lo lejes que esté el fantasma). La función: $150 / (dist + 0.5)$ produce una curva hiperbólica:
- Penalización agresiva en distancias cortas
- Penalización suave a distancias largas

El factor heurístico dos hace que pacman solo reaccione cuando los fantasmas ya están encima de él, pudiendo provocar siutaciones donde se den encerronas. El factor heurístico que proponemos hace que Pacman sienta contínuamente la presión de escapar y comience a alejarse de forma actica antes de que la situación sea crítica. 

**Justifiación**:
&rarr; La penalización solo se aplica a fantasmas **no** asustados (```python scaredTimer == 0 ```). La idea principal es que, un Pacman que sobrevive más tiempo tiene más oportunidades que comer _cocos_ y acumular puntuación. Huir activamente de los fantasmas evita situaciones críticas que no aniticipa el factor 2. Al final, un Pacman que se dedica a huir puede acabar ganando una partida puesto que mientras que huye puede ir comiendo _cocos_

- Peso elegido: 150. Un peso menor (ej. 50) no superaba la inercia del score base y Pacman seguía ignorando fantasmas lejanos; un peso mayor (ej. 300) hacía que Pacman huyera tanto que dejaba de comer. 150 equilibra supervivencia y comida

#### Heurística 4 - Food Clustering 
```python
if food:
    distances_to_food = sorted(
        manhattanDistance(pacman_pos, food_pos) for food_pos in food
    )
    nearest_cluster = distances_to_food[:5]
    avg_cluster_dist = sum(nearest_cluster) / len(nearest_cluster)
    score -= 0.4 * avg_cluster_dist
```
Esta heurísitca calcula la distancia media a los 5 _cocos_ más cercanos y resta 0.4 * esa media a la puntuación. Si los _cocos_ cercanos están agrupados, la distancia media es baja y la penalización es mínima, luego pacman tenderá a limpiar esa zona. Si los _cocos_ cercanos están dispersos, la penalización aumenta y pacman no tenderá a moverse a esa zona
- Penalización alta si los cocos cercanos están separados
- Penalización suave si los cocos cercanos están cercanos

El factor heurístico uno hace que pacman tienda al coco más cercano. Eso puede derivar en que pacman esté zigzagueando entre cocos aislados sin aprovechar agrupaciones. El factor que proponemos la dá una visión a pacman sobre la densidad de comida 

**Justifiación**:
&rarr; Acabar las partidas rápidamente minimiza el tiempo de exposición a los fantasmas y maximiza la puntuación final. Un pacman capaz de limpiar las zonas densas primero es capaz de limpiar el tablero de forma más eficiente y rápida que uno que persigue _cocos_ de uno en uno en extremos opuestos del mapa

- Peso elegido: 0.4. Las distancias de Manhattan en el mapa ``mediumClassic`` oscilan entre 1 y aprox(20), por lo que la contribución media de este factor es de entre -0.4 y -8 puntos, una influencia moderada que guía sin dominar la decisión.

#### Separación de agentes

Uno de los objetivos era ser capaz de comparar ambas versiones (2 factores heurísticos | 4 factores heurísitcos) sin tener que modificar código. Para llevar a cabo este objetivo, las distintas heurísticas se han separado en dos clases independientes:
- `NeuralAgent`: función de evaluación original. Contiene las dos heurísitcas básicas (sin modificar)
- `NeuralAgent2`: hereda de `NeuralAgent` y sobreescribe únicamente `evaluationFunction` incluyendo (a los dos factores originales) nuestros factores heurísticos propuestos

De esta forma, podemos ejecutar cada versión de forma independiente y comparar los resultados directamente:

**Ejecución:**

```bash
# Agente original (2 factores)
python pacman.py -p NeuralAgent -q -n 10

# Agente mejorado (4 factores)
python pacman.py -p NeuralAgentV2 -q -n 10

```
### Resultados esperados
Se espera que `NeuralAgent2` consiga un comportamiento diferente al original: tendendia a evitar sitaciones críticas y limpiar zonas con densidad alta de _cocos_. En cuanto a métricas, es posible que el winrate no mejore drásticamente respecto al original, ya que la red neuronal subyacente fue entrenada con datos del agente original y existe un desajuste entre el modelo y las nuevas heurísticas. Este desajuste se corrige en la Task 2 reentrenando el modelo con partidas generadas por ``NeuralAgentV2``


## Parte 2 - Entrenamiento de la Red Neuronal

## Parte 3 - AlphaBetaNeuralAgent

## Parte 4 - Resultados

## Conclusiones

## Comandos del Video
