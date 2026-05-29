# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import torch
import numpy as np
from net import PacmanNet, HIDDEN_SIZE
import os
from util import manhattanDistance
from game import Directions
import random, util
if not os.getenv('PACMAN_RANDOM'):
    random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState, Actions

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = os.getenv('AGENT_DEPTH', '2')):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

# <=====================================================================================================================>
# <=====================================================================================================================>
# <=====================================================================================================================>

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for Pacman with multiple ghosts
    """

    def _minimax(self, agentIndex, depth, gameState):
        """
        Recursive minimax function

        Args:
        - agentIndex: Current agent (0=Pacman, 1+=Ghosts)
        - depth: Current depth in the game tree
        - gameState: Current state of the game

        Returns:
        - Best evaluation score for this state
        """
        # Base case: terminal state or maximum depth reached
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # Pacman's turn (Maximizer)
        if agentIndex == 0:
            return self._maxValue(agentIndex, depth, gameState)
        # Ghost's turn (Minimizer)
        else:
            return self._minValue(agentIndex, depth, gameState)

    def _maxValue(self, agentIndex, depth, gameState):
        """
        Handles Pacman's moves (maximizing player)
        """
        v = float('-inf')  # Start with worst possible value
        legalActions = gameState.getLegalActions(agentIndex)

        # No legal actions available
        if not legalActions:
            return self.evaluationFunction(gameState)

        # Try each possible action and choose the best
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # After Pacman moves, first ghost plays (agent 1)
            v = max(v, self._minimax(1, depth, successor))
        return v

    def _minValue(self, agentIndex, depth, gameState):
        """
        Handles Ghost moves (minimizing players)
        """
        v = float('inf')  # Start with best possible value for Pacman
        legalActions = gameState.getLegalActions(agentIndex)

        # No legal actions available
        if not legalActions:
            return self.evaluationFunction(gameState)

        # Determine next agent and depth
        nextAgent = agentIndex + 1
        nextDepth = depth

        # If all ghosts have moved, return to Pacman and increment depth
        if nextAgent == gameState.getNumAgents():
            nextAgent = 0      # Back to Pacman
            nextDepth = depth + 1  # New ply begins

        # Try each possible action and choose the worst for Pacman
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self._minimax(nextAgent, nextDepth, successor))
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """
        # Main decision logic for Pacman
        bestAction = None
        bestScore = float('-inf')

        # Try each legal action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = self._minimax(1, 0, successor)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def _alphaBeta(self, agentIndex, depth, gameState, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self._maxValue(agentIndex, depth, gameState, alpha, beta)
        else:
            return self._minValue(agentIndex, depth, gameState, alpha, beta)

    def _maxValue(self, agentIndex, depth, gameState, alpha, beta):
        v = float('-inf')
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(gameState)

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self._alphaBeta(1, depth, successor, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def _minValue(self, agentIndex, depth, gameState, alpha, beta):
        v = float('inf')
        legalActions = gameState.getLegalActions(agentIndex)

        if not legalActions:
            return self.evaluationFunction(gameState)

        nextAgent = agentIndex + 1
        nextDepth = depth

        if nextAgent == gameState.getNumAgents():
            nextAgent = 0
            nextDepth = depth + 1

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self._alphaBeta(nextAgent, nextDepth, successor, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self._alphaBeta(1, 0, successor, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

class AlphaBetaRnAgent(AlphaBetaAgent):
    '''
    Minimax agent with alpha-beta pruning and randomized tie-breaking
    '''
    tolerance = 0.0

    def getAction(self, gameState):
        '''
        Returns the minimax action using self.depth and self.evaluationFunction
        '''
        bestActions = []
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self._alphaBeta(1, 0, successor, alpha, beta)

            if bestScore == float('-inf'):
                tolerance_window = 0.0
            else:
                tolerance_window = type(self).tolerance * abs(bestScore)

            if score > bestScore + tolerance_window:
                bestScore = score
                bestActions = [action]
            elif abs(score - bestScore) <= tolerance_window:
                bestActions.append(action)
                bestScore = max(bestScore, score)
            alpha = max(alpha, bestScore)

        return random.choice(bestActions)

def getMazeDistance(pos1: tuple[float, float], pos2: tuple[float, float], gameState: 'GameState') -> float:
    '''
    Calcula la distancia real en el laberinto entre dos puntos usando el algoritmo
    de búsqueda en anchura (BFS) para sortear las paredes.

    :param pos1: Coordenadas (x, y) del punto de origen.
    :param pos2: Coordenadas (x, y) del punto de destino.
    :param gameState: Estado actual de la partida, necesario para extraer la matriz de muros.
    :return: Distancia mínima en pasos. Devuelve float('inf') si el destino es inalcanzable.
    '''
    # Redondeo de coordenadas al centro de la celda más cercana
    x1, y1 = int(pos1[0] + 0.5), int(pos1[1] + 0.5)
    x2, y2 = int(pos2[0] + 0.5), int(pos2[1] + 0.5)

    # Retorno temprano si origen y destino coinciden
    if (x1, y1) == (x2, y2):
        return 0.0

    # Inicialización de la cola BFS y el conjunto de nodos visitados
    queue: list[tuple[int, int, int]] = [(x1, y1, 0)]
    visited: set[tuple[int, int]] = {(x1, y1)}
    walls = gameState.getWalls()

    # Exploración iterativa nivel por nivel
    while queue:
        x, y, dist = queue.pop(0)

        # Condición de éxito: se ha alcanzado la celda destino
        if (x, y) == (x2, y2):
            return float(dist)

        # Expansión del nodo hacia las 4 direcciones ortogonales
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Validación de límites del mapa, ausencia de pared y nodo no explorado
            if 0 <= nx < walls.width and 0 <= ny < walls.height:
                if not walls[nx][ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

    # Caso de fallo: no existe ningún camino físico que conecte los puntos
    return float('inf')

class HibridAgent(AlphaBetaAgent):
    '''
    Agente basado en dos estados (Pacifico, Supervivencia) dictados por una simple heuristica
    booleana de riesgo. El modo pacifico esta basado en una serie de heuristicas topologicas para
    ir comiendose puntos de manera eficiente y el modo supervivencia usa una busqueda Alpha-Beta
    con una heuristica (funcion de evaluacion) basada en maximizar la supervivencia.
    '''

    def __init__(self, depth: str = '4') -> None:
        '''
        Inicializa el agente híbrido delegando en la clase base y sobrescribiendo la evaluación.

        :param depth: Profundidad de búsqueda para el árbol Minimax.
        '''
        super().__init__('scoreEvaluationFunction', depth) # Luego sobrescribimos eval func
        self.evaluationFunction = self.evaluationFunction_local

    def _init_matrices(self, gameState: 'GameState') -> None:
        '''
        Precomputa la topología del laberinto usando BFS para generar matrices de distancias
        reales (teniendo en cuenta laberinto), identificar intersecciones y calcular la geometría de los nodos.

        :param gameState: Estado actual del juego.
        '''
        # Patrón Singleton para evitar recalcular en cada turno
        if getattr(self, '_initialized', False): return None

        walls = gameState.getWalls()
        self.width, self.height = walls.width, walls.height

        # Filtra solo las casillas transitables
        valid_positions: list[tuple[int, int]] = [(x, y) for x in range(self.width) for y in range(self.height) if not walls[x][y]]

        self.distance_matrix: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

        # BFS desde cada casilla transitable a todas las demás
        for start_pos in valid_positions:
            self.distance_matrix[(start_pos, start_pos)] = 0
            queue: list[tuple[int, int, int]] = [(start_pos[0], start_pos[1], 0)]
            visited: set[tuple[int, int]] = {start_pos}
            while queue:
                cx, cy, dist = queue.pop(0)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and not walls[nx][ny]:
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            self.distance_matrix[(start_pos, (nx, ny))] = dist + 1
                            queue.append((nx, ny, dist + 1))

        self.intersections: set[tuple[int, int]] = set()

        # Identificación de nodos de decisión (aquellos con >= 3 salidas)
        for pos in valid_positions:
            x, y = pos
            open_neighbors = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                 if 0 <= x+dx < self.width and 0 <= y+dy < self.height and not walls[x+dx][y+dy])
            if open_neighbors >= 3: self.intersections.add(pos)

        self.node_geometry: dict[tuple[int, int], tuple[int, int, int]] = {}

        # Calcula la distancia desde cada posición a sus intersecciones más cercanas
        for pos in valid_positions:
            queue = [(pos[0], pos[1], 0)]
            visited = {pos}
            exits: list[tuple[tuple[int, int], int]] = []
            while queue:
                cx, cy, dist = queue.pop(0)
                if (cx, cy) in self.intersections and (cx, cy) != pos:
                    exits.append(((cx, cy), dist))
                    continue
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and not walls[nx][ny]:
                        if (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))

            distances: list[int] = [e[1] for e in exits]
            if distances:
                self.node_geometry[pos] = (min(distances), max(distances), sum(distances))
            else:
                self.node_geometry[pos] = (0, 0, 0)

        # Initialization guard
        self._initialized = True

    def _getMazeDistance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        '''
        Recupera la distancia real de laberinto precomputada entre dos puntos.

        :param p1: Coordenadas (x, y) del primer punto.
        :param p2: Coordenadas (x, y) del segundo punto.
        :return: Distancia en pasos, o float('inf') si es inalcanzable.
        '''
        pos1 = (int(p1[0]+0.5), int(p1[1]+0.5))
        pos2 = (int(p2[0]+0.5), int(p2[1]+0.5))
        return self.distance_matrix.get((pos1, pos2), float('inf'))

    def _is_ghost_moving_away(self, pacman_pos: tuple[float, float], ghost_state: 'GhostState', walls: 'Grid') -> bool:
        '''
        Comprueba si el vector de movimiento de un fantasma incrementa su distancia a Pacman.

        :param pacman_pos: Coordenadas actuales de Pacman.
        :param ghost_state: Objeto de estado del fantasma a evaluar.
        :param walls: Matriz de booleanos de las paredes.
        :return: True si el fantasma se aleja en el siguiente frame, False de lo contrario.
        '''
        g_pos = ghost_state.getPosition()
        g_dir = ghost_state.getDirection()

        if g_dir == 'Stop': return False

        vectors: dict[str, tuple[int, int]] = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}
        dx, dy = vectors.get(g_dir, (0, 0))
        next_x, next_y = int(g_pos[0] + dx), int(g_pos[1] + dy)

        # Previene errores si el motor intenta simular un movimiento contra una pared
        if walls[next_x][next_y]: return False

        dist_now = self._getMazeDistance(pacman_pos, g_pos)
        dist_next = self._getMazeDistance(pacman_pos, (next_x, next_y))

        return dist_next > dist_now

    # NUEVO: Calculador de Inercia Estricta (Bloqueo de giro de 180º)
    def _get_ghost_effective_dist(self, ghost_state: 'GhostState', target_pos: tuple[float, float], walls: 'Grid') -> float:
        '''
        Calcula la distancia inercial estricta asumiendo que los fantasmas no pueden
        girar 180 grados a menos que lleguen a un callejón sin salida.

        :param ghost_state: Estado actual del fantasma.
        :param target_pos: Coordenadas del objetivo (comida o cápsula).
        :param walls: Matriz de paredes del laberinto.
        :return: Distancia obligada en pasos.
        '''
        g_pos = ghost_state.getPosition()
        g_dir = ghost_state.getDirection()

        pos1 = (int(g_pos[0]+0.5), int(g_pos[1]+0.5))
        target = (int(target_pos[0]+0.5), int(target_pos[1]+0.5))

        base_dist = self._getMazeDistance(pos1, target)
        if g_dir == 'Stop': return base_dist

        vectors: dict[str, tuple[int, int]] = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}
        dx, dy = vectors.get(g_dir, (0, 0))
        nx, ny = pos1[0] + dx, pos1[1] + dy

        # Si choca de frente, girará, así que la distancia base aplica
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or walls[nx][ny]:
            return base_dist

        dist_next = self._getMazeDistance((nx, ny), target)
        if dist_next <= base_dist:
            return base_dist # Se acerca de forma natural

        # El fantasma se aleja. Trazamos su ruta física obligatoria sin girar 180º.
        cx, cy = nx, ny
        prev_x, prev_y = pos1[0], pos1[1]
        steps = 1

        while True:
            open_neighbors: list[tuple[int, int]] = []
            for dxx, dyy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                test_x, test_y = cx + dxx, cy + dyy
                if 0 <= test_x < self.width and 0 <= test_y < self.height and not walls[test_x][test_y]:
                    open_neighbors.append((test_x, test_y))

            # Si hay intersección o callejón ciego, las reglas del juego le permiten girar
            if len(open_neighbors) >= 3 or len(open_neighbors) <= 1:
                break

            # Pasillo: Sigue adelante por obligación
            moved = False
            for n_pos in open_neighbors:
                if n_pos != (prev_x, prev_y):
                    prev_x, prev_y = cx, cy
                    cx, cy = n_pos
                    steps += 1
                    moved = True
                    break
            if not moved: break

        return steps + self._getMazeDistance((cx, cy), target)

    def _get_mst_cost(self, pacman_pos: tuple[float, float], food_list: list[tuple[float, float]]) -> float:
        '''
        Aproximación Greedy al problema del viajante usando el algoritmo de Prim para el
        árbol de expansión mínima (MST) sobre los puntos de comida restantes.

        :param pacman_pos: Coordenadas iniciales de Pacman.
        :param food_list: Lista de coordenadas de la comida activa.
        :return: Coste total estimado del MST.
        '''
        if not food_list: return 0
        nodes = [pacman_pos] + food_list
        unvisited = set(nodes)
        unvisited.remove(pacman_pos)
        min_dist_to_tree = {node: self._getMazeDistance(pacman_pos, node) for node in unvisited}
        mst_cost: float = 0

        while unvisited:
            closest = min(unvisited, key=lambda n: min_dist_to_tree[n])
            cost = min_dist_to_tree[closest]
            mst_cost += cost
            unvisited.remove(closest)

            for v in unvisited:
                d = self._getMazeDistance(closest, v)
                if d < min_dist_to_tree[v]:
                    min_dist_to_tree[v] = d

        return mst_cost

    def _is_state_safe(self, state: 'GameState') -> bool:
        '''
        Oráculo que determina el cambio de contexto entre modo Pacífico (Safe)
        y modo Pánico (Caution) basándose en proximidad e inercia enemiga.

        :param state: Estado a evaluar.
        :return: True si no hay amenaza inminente, False para activar evasión.
        '''
        pacman_pos = state.getPacmanPosition()
        walls = state.getWalls()

        # Filtra fantasmas ignorando a los asustados que no suponen una amenaza temporal
        active_ghosts = [g for g in state.getGhostStates() if g.scaredTimer <= self._getMazeDistance(pacman_pos, g.getPosition())]

        if not active_ghosts: return True

        # Pánico incondicional por vecindad inmediata
        if min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in active_ghosts) <= 3:
            return False

        # Verifica si los fantasmas se están acercando físicamente
        threats = [g for g in active_ghosts if not self._is_ghost_moving_away(pacman_pos, g, walls)]
        if not threats: return True

        # Radio de seguridad ampliado frente a atacantes directos
        if min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in threats) <= 5:
            return False

        return True

    def _pacificEvaluationFunction(self, state: 'GameState') -> float:
        '''
        Heurística orientada a la recolección óptima. Usada únicamente cuando el Oráculo
        dictamina que el área está segura. Focalizada en limpiar endpoints y optimizar MST.

        :param state: Nodo hoja generado a evaluar.
        :return: Puntuación estática del estado.
        '''
        # FIX 1: Muerte como infinito negativo
        if state.isLose(): return float('-inf')
        if state.isWin(): return float('inf')

        pacman_pos = state.getPacmanPosition()
        active_ghosts = [g for g in state.getGhostStates() if g.scaredTimer == 0]
        scared_ghosts = [g for g in state.getGhostStates() if g.scaredTimer > 0]

        if active_ghosts:
            min_active_dist = min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in active_ghosts)
            if min_active_dist <= 1:
                return float('-inf')

        score = state.getScore() * 1000.0

        for sg in scared_ghosts:
            if self._getMazeDistance(pacman_pos, sg.getPosition()) <= 1:
                # FIX 1: Chocar con asustado resta, pero salva del '-inf'
                score -= 100000.0

        # Voronoi parcial (límite 40 nodos) para medir el volumen espacial de escape
        walls = state.getWalls()
        safe_territory: int = 0
        queue = [(pacman_pos[0], pacman_pos[1], 0)]
        visited = {pacman_pos}

        while queue and safe_territory < 40:
            cx, cy, p_dist = queue.pop(0)
            safe_territory += 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not walls[nx][ny]:
                    if (nx, ny) not in visited:
                        node_safe = True
                        for g in active_ghosts:
                            # Previsión para colisiones adelantadas
                            if self._getMazeDistance(g.getPosition(), (nx, ny)) <= p_dist + 2:
                                node_safe = False
                                break
                        if node_safe:
                            visited.add((nx, ny))
                            queue.append((nx, ny, p_dist + 1))

        # Restricción topológica para evitar entrar voluntariamente en cuellos de botella
        if safe_territory < 15:
            score -= 900000.0

        capsules = state.getC

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {'cuda' if torch.cuda.is_available() else 'cpu'}!')
        self.load_model(model_path)

        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }

        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}

        # Contador de movimientos
        self.move_count = 0

        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False

            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']

            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, HIDDEN_SIZE, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación

            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height

        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)

        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1

        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2

        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3

        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal

        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5

        # Normalizar
        numeric_map = numeric_map / 6.0

        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0

        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)

        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Obtener acciones legales
        legal_actions = state.getLegalActions()

        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()

        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()

        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)

        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca

        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100

        return score + neural_score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1

        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)

        # Obtener acciones legales
        legal_actions = state.getLegalActions()

        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))

        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)

        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)

        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            #horror_score = self.getHorrorScore(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = (eval_score) + (neural_score)

            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50

            successors.append((action, combined_score))

        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)

        # Devolver la mejor acción
        return successors[0][0]

    ###########################################################################
    # Salas
    ###########################################################################

    def getHorrorScore(self, state):
        """
        Calcula un valor basado puramente en el miedo a los fantasmas.
        A mayor distancia, mayor puntuación.
        """
        pacman_pos = state.getPacmanPosition()
        ghost_states = state.getGhostStates()
        horror_score = 0

        for ghost_state in ghost_states:
            # Solo tenemos miedo si el fantasma NO está asustado
            if ghost_state.scaredTimer == 0:
                dist = manhattanDistance(pacman_pos, ghost_state.getPosition())

                # 1. Penalización masiva por cercanía inmediata (miedo al contacto)
                # Si la distancia es 1 o 2, restamos un valor enorme.
                if dist < 3:
                    horror_score -= 5000 / (dist + 0.1)
                    return horror_score

                # 2. Bonificación por distancia (incentivo de huida)
                # Cada punto de distancia de Manhattan suma agresivamente.
                horror_score += dist * 5000

        return horror_score

class ScaredyNeuralAgent(NeuralAgent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego. Este en especifico tiene
    una funcion de evaluacion que prioriza huir de los fantasmas.
    """
    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales (huir).
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0

        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)

        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Obtener acciones legales
        legal_actions = state.getLegalActions()

        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()

        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()

        # Factor 1: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

            # Si el fantasma esta asustado lo ignoramos
            if not ghost_state.scaredTimer > ghost_distance:
                # Cuanto mas lejos mejor
                score += ghost_distance * 20
                # Si esta muy cerca, fatal
                if ghost_distance <= 3:
                    # Penalizacion gradual para evitar acercamiento
                    score -= 200 * (4 - ghost_distance)  # Gran penalización por estar demasiado cerca

        # Factor 2: Distancia a la comida más cercana
        if food:
            # Para desempatar
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += (1.0 / (min_food_distance + 1)) * 20

        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 15

        return score + neural_score

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)

