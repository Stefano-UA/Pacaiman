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

from typing import override

import torch
import numpy as np
from net import PacmanNet, HIDDEN_SIZE
import os
from util import manhattanDistance
from game import Directions
import random, util
if not os.getenv('PACMAN_RANDOM'):
    random.seed(int(os.getenv('SEED', 42)))  # For reproducibility
from game import Agent
from pacman import GameState

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
# <== Stefano ==========================================================================================================>
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
    '''
    Your minimax agent with alpha-beta pruning (question 3)
    '''
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

        # Move Ordering O(1): Retrasar 'Stop' para evaluar acciones útiles primero
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
            legalActions.append('Stop')

        for action in legalActions:
            # Generación perezosa: solo se clona si no ha habido poda
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

    def getAction(self, gameState: 'GameState'):
        '''
        Returns the minimax action using self.depth and self.evaluationFunction
        '''
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        legalActions = gameState.getLegalActions(0)

        if 'Stop' in legalActions:
            legalActions.remove('Stop')
            legalActions.append('Stop')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self._alphaBeta(1, 0, successor, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

class AlphaBetaRnAgent(AlphaBetaAgent):
    '''
    Minimax agent with alpha-beta pruning and randomized selection when there are ties between options.
    '''

    # Tolerance for whats considered a tie
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
    if (x1, y1) == (x2, y2): return 0.0

    # Inicialización de la cola BFS y el conjunto de nodos visitados
    queue: list[tuple[int, int, int]] = [(x1, y1, 0)]
    visited: set[tuple[int, int]] = {(x1, y1)}
    walls = gameState.getWalls()

    # Exploración iterativa nivel por nivel
    while queue:
        x, y, dist = queue.pop(0)
        # Se ha alcanzado la celda destino
        if (x, y) == (x2, y2): return float(dist)
        # Expansión del nodo hacia las 4 direcciones ortogonales
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            # Validación de límites del mapa, ausencia de pared y nodo no explorado
            if 0 <= nx < walls.width and 0 <= ny < walls.height:
                if not walls[nx][ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))

    # No existe ningún camino físico que conecte los puntos
    return float('inf')

class HibridAgent(AlphaBetaAgent):
    '''
    Agente hibrido de dos estados (Pacifico, Supervivencia) controlado por una evaluacion de riesgo.
    - Modo Pacifico: Busqueda Greedy guiada por heuristicas topologicas (MST, Voronoi).
    - Modo Supervivencia: Busqueda adversaria Alpha-Beta para maximizar tiempo de vida.
    '''
    def __init__(self, depth: str = '4') -> None:
        '''
        Inicializa el agente y enlaza la funcion de evaluacion local para la poda Alpha-Beta.

        :param depth: Profundidad maxima del arbol de busqueda Minimax.
        '''
        # Inicializa la clase base y sobrescribe la heuristica de evaluacion
        super().__init__('scoreEvaluationFunction', depth)
        self.evaluationFunction = self.evaluationFunction_local

    def _init_matrices(self, gameState: 'GameState') -> None:
        '''
        Precomputa la topologia estatica del laberinto en la primera llamada.
        Genera matrices de distancias, detecta intersecciones y cachea
        rutas obligatorias para evaluar la inercia en O(1).

        :param gameState: Estado fisico inicial para extraer la cuadricula de muros.
        '''
        # Patron Singleton: evita recalcular la topologia si ya esta inicializada
        if getattr(self, '_initialized', False): return None

        walls = gameState.getWalls()
        self.width, self.height = walls.width, walls.height
        # Extrae las posiciones transitables del laberinto
        valid_positions: list[tuple[int, int]] = [(x, y) for x in range(self.width) for y in range(self.height) if not walls[x][y]]

        self.distance_matrix: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        # 1. Matriz de adyacencia completa mediante BFS desde cada casilla
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
        # 2. Identifica intersecciones (nodos con grado topologico >= 3)
        for pos in valid_positions:
            x, y = pos
            open_neighbors = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                 if 0 <= x+dx < self.width and 0 <= y+dy < self.height and not walls[x+dx][y+dy])
            if open_neighbors >= 3: self.intersections.add(pos)

        self.node_geometry: dict[tuple[int, int], tuple[int, int, int]] = {}
        # 3. Geometria de pasillos: calcula distancias a las intersecciones mas cercanas
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

        # 4. Cache de inercia O(1): Mapea (casilla_previa, casilla_actual) a su salida forzada
        self.directed_paths: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int], int, dict[tuple[int, int], int]]] = {}
        for p in valid_positions:
            neighbors: list[tuple[int, int]] = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = p[0] + dx, p[1] + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not walls[nx][ny]:
                    neighbors.append((nx, ny))

            # Calcula la ruta inercial obligatoria del fantasma desde cada origen
            for prev_pos in neighbors:
                cx, cy = p
                px, py = prev_pos
                steps: int = 1
                path_dict: dict[tuple[int, int], int] = {(cx, cy): 0}
                # Simula el avance inercial por el pasillo
                while True:
                    c_neighbors: list[tuple[int, int]] = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nnx, nny = cx + dx, cy + dy
                        if 0 <= nnx < self.width and 0 <= nny < self.height and not walls[nnx][nny]:
                            c_neighbors.append((nnx, nny))

                    # Si alcanza interseccion o callejon, guarda la ruta en cache y termina
                    if len(c_neighbors) >= 3 or len(c_neighbors) <= 1:
                        self.directed_paths[(prev_pos, p)] = ((cx, cy), (px, py), steps, path_dict)
                        break

                    # Avance determinista: selecciona la salida distinta a la de entrada
                    next_pos = c_neighbors[0] if c_neighbors[0] != (px, py) else c_neighbors[1]
                    px, py = cx, cy
                    cx, cy = next_pos
                    path_dict[(cx, cy)] = steps
                    steps += 1

        self._initialized = True

    def _getMazeDistance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        '''
        Consulta la distancia precomputada O(1).

        :param p1: Coordenadas de origen.
        :param p2: Coordenadas de destino.
        :return: Distancia en pasos, o infinito si es inalcanzable.
        '''
        # Redondea coordenadas al centro de la celda
        pos1 = (int(p1[0]+0.5), int(p1[1]+0.5))
        pos2 = (int(p2[0]+0.5), int(p2[1]+0.5))
        return self.distance_matrix.get((pos1, pos2), float('inf'))

    def _is_ghost_moving_away(self, pacman_pos: tuple[float, float], ghost_state: 'GhostState', walls: 'Grid') -> bool:
        '''
        Compara la distancia actual del fantasma con la distancia tras proyectar su vector un frame.

        :param pacman_pos: Posicion de Pacman.
        :param ghost_state: Estado interno del fantasma.
        :param walls: Matriz booleana de muros.
        :return: True si el fantasma aumenta su distancia relativa.
        '''
        g_pos = ghost_state.getPosition()
        g_dir = ghost_state.getDirection()
        if g_dir == 'Stop': return False

        # Proyecta el vector de movimiento un frame
        vectors: dict[str, tuple[int, int]] = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}
        dx, dy = vectors.get(g_dir, (0, 0))
        next_x, next_y = int(g_pos[0] + dx), int(g_pos[1] + dy)
        # Retorna False si el movimiento choca contra un muro
        if walls[next_x][next_y]: return False

        # Compara distancia actual y proyectada para determinar alejamiento
        dist_now = self._getMazeDistance(pacman_pos, g_pos)
        dist_next = self._getMazeDistance(pacman_pos, (next_x, next_y))
        return dist_next > dist_now

    def _get_ghost_effective_dist(self, ghost_state: 'GhostState', target_pos: tuple[float, float], walls: 'Grid') -> float:
        '''
        Calcula la cota inferior estricta de distancia a un objetivo forzando el cumplimiento
        de la regla de inercia (no 180). Utiliza cache precomputada.

        :param ghost_state: Estado del fantasma a simular.
        :param target_pos: Coordenadas del objetivo de intercepcion.
        :param walls: Matriz booleana de muros.
        :return: Pasos minimos garantizados hasta el objetivo.
        '''
        g_pos = ghost_state.getPosition()
        g_dir = ghost_state.getDirection()
        pos = (int(g_pos[0] + 0.5), int(g_pos[1] + 0.5))
        target = (int(target_pos[0] + 0.5), int(target_pos[1] + 0.5))

        base_dist = self._getMazeDistance(pos, target)
        if g_dir == 'Stop': return base_dist

        vectors: dict[str, tuple[int, int]] = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}
        dx, dy = vectors.get(g_dir, (0, 0))
        nx, ny = pos[0] + dx, pos[1] + dy

        # Aplica distancia base si el movimiento choca contra un muro
        if (nx < 0) or (nx >= self.width) or (ny < 0) or (ny >= self.height) or walls[nx][ny]:
            return base_dist

        # Aplica distancia base si el fantasma ya se acerca al objetivo
        dist_next = self._getMazeDistance((nx, ny), target)
        if dist_next <= base_dist:
            return base_dist

        # Recupera la topologia precalculada del pasillo O(1)
        end_pos, prev_of_end, steps_to_end, path_dict = self.directed_paths[(pos, (nx, ny))]

        # Retorna distancia exacta si el objetivo se encuentra en el pasillo forzado
        if target in path_dict:
            return float(1 + path_dict[target])

        open_neighbors: list[tuple[int, int]] = []
        for dxx, dyy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            tx, ty = end_pos[0] + dxx, end_pos[1] + dyy
            if 0 <= tx < self.width and 0 <= ty < self.height and not walls[tx][ty]:
                open_neighbors.append((tx, ty))

        # Calcula distancia con giro de 180 si es un callejon sin salida
        if len(open_neighbors) <= 1:
            return float(steps_to_end + self._getMazeDistance(end_pos, target))

        # Evalua las ramas de la interseccion omitiendo la de procedencia
        valid_next_steps = [n for n in open_neighbors if n != prev_of_end]
        min_branch_dist = float('inf')

        for next_step in valid_next_steps:
            b_end_pos, b_prev_of_end, b_steps, b_path_dict = self.directed_paths[(end_pos, next_step)]
            # Comprueba si el objetivo esta en la rama evaluada
            if target in b_path_dict:
                dist_via_this_branch = float(1 + b_path_dict[target])
            else:
                dist_via_this_branch = float(b_steps + self._getMazeDistance(b_end_pos, target))

            if dist_via_this_branch < min_branch_dist:
                min_branch_dist = dist_via_this_branch

        return float(steps_to_end + min_branch_dist)

    def _get_mst_cost(self, pacman_pos: tuple[float, float], food_list: list[tuple[float, float]]) -> float:
        '''
        Calcula el Arbol de Expansion Minima (Algoritmo de Prim).

        :param pacman_pos: Nodo raiz.
        :param food_list: Nodos a conectar.
        :return: Coste del MST.
        '''
        if not food_list: return 0
        nodes = [pacman_pos] + food_list
        unvisited = set(nodes)
        unvisited.remove(pacman_pos)
        # Cachea distancias desde el arbol a los nodos no visitados
        min_dist_to_tree = {node: self._getMazeDistance(pacman_pos, node) for node in unvisited}
        mst_cost: float = 0

        while unvisited:
            # Añade el nodo mas cercano al MST actual
            closest = min(unvisited, key=lambda n: min_dist_to_tree[n])
            cost = min_dist_to_tree[closest]
            mst_cost += cost
            unvisited.remove(closest)

            # Actualiza distancias contra el nuevo nodo del MST
            for v in unvisited:
                d = self._getMazeDistance(closest, v)
                if d < min_dist_to_tree[v]:
                    min_dist_to_tree[v] = d

        return mst_cost

    def _is_state_safe(self, state: 'GameState') -> bool:
        '''
        Determina si el entorno permite exploracion Greedy en lugar de evaluacion Minimax.

        :param state: Estado global del tablero.
        :return: True si no existe amenaza inminente.
        '''
        pacman_pos = state.getPacmanPosition()
        walls = state.getWalls()

        # Filtra fantasmas asustados que no representan amenaza temporal
        active_ghosts = [g for g in state.getGhostStates() if g.scaredTimer <= self._getMazeDistance(pacman_pos, g.getPosition())]
        if not active_ghosts: return True

        # Retorna False si un fantasma activo esta a distancia <= 3
        if min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in active_ghosts) <= 3: return False

        # Evalua fantasmas que se acercan inercialmente
        threats = [g for g in active_ghosts if not self._is_ghost_moving_away(pacman_pos, g, walls)]
        if not threats: return True

        # Retorna False si los fantasmas que se acercan estan a distancia <= 5
        if min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in threats) <= 5: return False

        return True

    def _pacificEvaluationFunction(self, state: 'GameState') -> float:
        '''
        Heuristica de recoleccion (Modo Greedy). Prioriza recoleccion topologica.

        :param state: Estado a evaluar.
        :return: Score numerico de calidad del nodo.
        '''
        # Evaluacion de estados terminales
        if state.isLose(): return float('-inf')
        if state.isWin(): return float('inf')

        pacman_pos = state.getPacmanPosition()
        active_ghosts = [g for g in state.getGhostStates() if g.scaredTimer == 0]
        scared_ghosts = [g for g in state.getGhostStates() if g.scaredTimer > 0]

        # Penalizacion letal ante colision inminente
        if active_ghosts:
            min_active_dist = min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in active_ghosts)
            if min_active_dist <= 1:
                return float('-inf')

        score = state.getScore() * 1000.0

        # Penalizacion por colision con fantasma asustado
        for sg in scared_ghosts:
            if self._getMazeDistance(pacman_pos, sg.getPosition()) <= 1:
                score -= 100000.0

        # Voronoi in-situ mediante BFS para detectar rutas bloqueadas temporalmente
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
                            # Cota topologica con margen de +2
                            if self._getMazeDistance(g.getPosition(), (nx, ny)) <= p_dist + 2:
                                node_safe = False
                                break
                        if node_safe:
                            visited.add((nx, ny))
                            queue.append((nx, ny, p_dist + 1))

        # Penalizacion por falta de territorio seguro
        if safe_territory < 15:
            score -= 900000.0

        capsules = state.getCapsules()
        score += len(capsules) * 5000.0

        food_list = state.getFood().asList()
        if food_list:
            min_food_dist = min(self._getMazeDistance(pacman_pos, f) for f in food_list)
            score += 200.0 / (min_food_dist + 0.1)

            # Penalizacion de nodos aislados (endpoints)
            endpoints: int = 0
            for f in food_list:
                neighbors = sum(1 for other in food_list if f != other and self._getMazeDistance(f, other) <= 2)
                if neighbors == 0:
                    endpoints += 3
                elif neighbors == 1:
                    endpoints += 1

            score -= endpoints * 2000.0
            # Incentiva recoleccion segun el coste MST
            mst_cost = self._get_mst_cost(pacman_pos, food_list)
            score -= mst_cost * 50.0

        return score

    def evaluationFunction_local(self, state: 'GameState') -> float:
        '''
        Heuristica de supervivencia estricta (Modo Minimax).
        Penaliza posiciones estructuralmente debiles.

        :param state: Nodo hoja del arbol de busqueda.
        :return: Score numerico de invulnerabilidad.
        '''
        if state.isLose(): return float('-inf')
        if state.isWin(): return float('inf')

        pacman_pos = state.getPacmanPosition()
        active_ghosts = [g for g in state.getGhostStates() if g.scaredTimer == 0]
        walls = state.getWalls()

        # Modo recoleccion prioritaria si no hay fantasmas activos
        if not active_ghosts:
            return 500000.0 + state.getScore() * 10.0

        # Penalizacion inverso-cuadratica por proximidad a fantasmas
        min_g_dist = min(self._getMazeDistance(pacman_pos, g.getPosition()) for g in active_ghosts)
        score = -500.0 / (min_g_dist ** 2 + 0.1)
        score += state.getScore() * 500.0

        # Voronoi proyectado para nodos futuros en Minimax
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
                            # Cota topologica con margen de +1
                            if self._getMazeDistance(g.getPosition(), (nx, ny)) <= p_dist + 1:
                                node_safe = False
                                break
                        if node_safe:
                            visited.add((nx, ny))
                            queue.append((nx, ny, p_dist + 1))

        is_trapped = (safe_territory < 15)

        # Penaliza encierros en el arbol Minimax
        if is_trapped:
            score -= 900000.0
            score += min_g_dist * 50.0
        else:
            score += safe_territory * 2000.0

        food_list = state.getFood().asList()
        if food_list:
            min_food_dist = min(self._getMazeDistance(pacman_pos, f) for f in food_list)
            score -= min_food_dist * 10.0

        capsules = state.getCapsules()
        override = False

        # Midgame Rush: prioriza capsulas seguras
        if capsules:
            closest_cap_dist = min([self._getMazeDistance(pacman_pos, c) for c in capsules])
            closest_ghost_to_cap = min([self._getMazeDistance(g.getPosition(), c) for c in capsules for g in active_ghosts])
            if closest_cap_dist < closest_ghost_to_cap:
                override = True
                score += 100000.0 / (closest_cap_dist + 1)
                if is_trapped:
                    score += 900000.0

        # Endgame Rush: simulacion de limpieza final
        if not override and food_list and len(food_list) <= 10:
            curr_pos = pacman_pos
            time_elapsed: int = 0
            safe_path = True
            unvisited = set(food_list)
            while unvisited:
                # Selecciona nodo mas cercano
                next_food = min(unvisited, key=lambda f: self._getMazeDistance(curr_pos, f))
                dist = self._getMazeDistance(curr_pos, next_food)
                time_elapsed += int(dist)
                curr_pos = next_food

                # Verifica colision futura usando distancia inercial del fantasma
                for g in active_ghosts:
                    g_eff_dist = self._get_ghost_effective_dist(g, next_food, walls)
                    if g_eff_dist <= time_elapsed + 1:
                        safe_path = False
                        break
                if not safe_path: break
                unvisited.remove(next_food)

            # Activa Endgame Rush si la ruta simulada es segura
            if safe_path:
                override = True
                mst_cost = self._get_mst_cost(pacman_pos, food_list)
                score += 9999999.0 / (mst_cost + 1)
                if is_trapped:
                    score += 900000.0

        # Penaliza posicionamiento en pasillos largos o cerrados
        if not override and not is_trapped and min_g_dist > 3:
            nearest, farthest, length = self.node_geometry.get(pacman_pos, (0, 0, 0))
            score -= length * 200.0
            score -= nearest * 50.0

        return score

    def getAction(self, gameState: 'GameState') -> str:
        '''
        Selector raiz de acciones. Selecciona evaluacion Greedy o Alpha-Beta segun riesgo.

        :param gameState: Estado actual inalterado.
        :return: Accion valida (e.g. 'North', 'Stop').
        '''
        self._init_matrices(gameState)
        legal_actions = gameState.getLegalActions(0)

        # Accion fallback: selecciona movimiento que maximiza distancia a enemigos en caso de muerte inevitable
        best_fallback = 'Stop'
        max_dist_to_death: float = -1
        for a in legal_actions:
            succ = gameState.generateSuccessor(0, a)
            p_pos = succ.getPacmanPosition()
            active_ghosts = [g for g in succ.getGhostStates() if g.scaredTimer == 0]
            if not active_ghosts:
                best_fallback = a
                break
            min_dist = min([self._getMazeDistance(p_pos, g.getPosition()) for g in active_ghosts])
            if min_dist > max_dist_to_death:
                max_dist_to_death = min_dist
                best_fallback = a

        fallback_action = best_fallback

        # Ejecucion Pacifico: busqueda Greedy a profundidad 1
        if self._is_state_safe(gameState):
            best_score = float('-inf')
            best_action = fallback_action
            for a in legal_actions:
                succ = gameState.generateSuccessor(0, a)
                score = self._pacificEvaluationFunction(succ)

                # Penalizacion por inaccion ('Stop')
                if a == 'Stop': score -= 15.0

                if score > best_score:
                    best_score = score
                    best_action = a
            return best_action

        # Ejecucion Supervivencia: Minimax Alpha-Beta
        else:
            best_score = float('-inf')
            best_action = fallback_action
            alpha, beta = float('-inf'), float('inf')

            for a in legal_actions:
                succ = gameState.generateSuccessor(0, a)
                score = self._alphaBeta(1, 0, succ, alpha, beta)

                if a == 'Stop': score -= 15.0

                if score > best_score:
                    best_score = score
                    best_action = a

                # Actualiza cota Alpha con el mejor score
                alpha = max(alpha, best_score)
            return best_action

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

    def traditionalEvaluation(self, state) -> float:
        '''
        Devuelve la parte del score de las heuristicas
        '''
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

        return score

    def neuralEvaluation(self, state) -> float:
        '''
        Devuelve la parte del score de la red neuronal.

        Se usa para tener en cuenta, segun la red neuronal, que tan buenas son las posibles acciones del estado actual.
        '''
        if self.model is None: return 0

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

        '''
        Al sumar las probabilidades softmax de todas las acciones legales (cuya suma siempre tiende a 1.0),
        solo estamos inyectando un sesgo constante de aprox 100 puntos a cualquier estado futuro, anulando por completo
        caulquier diferenciacion entre estados. Por eso este codigo lo quitamos.
        '''
        #score = 0
        #for i, action in enumerate(self.idx_to_action.values()):
        #    if action in legal_actions:
        #        score += probabilities[i] * 100

        # Combinar la puntuación heurística con la confianza de la red
        legal_probs = [probabilities[i] for i, action in enumerate(self.idx_to_action.values()) if action in legal_actions]
        score = max(legal_probs) * 100.0 if legal_probs else 0.0

        return score

    @override
    def evaluationFunction(self, state) -> float:
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        return self.traditionalEvaluation(state) + self.neuralEvaluation(state)

    @override
    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1

        # Si no hay modelo, salir
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()

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
        # REMOVED ya que hace que pacman se suicide con una probabilidad

        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score

            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50

            successors.append((action, combined_score))

        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)

        # Devolver la mejor acción
        return successors[0][0]

###########################################################################
# Alfonso
###########################################################################

class NeuralAgent2(NeuralAgent):
    """
    Versión de NeuralAgnet mejorada -> Usa 4 factores heurísiticos:
    2 originales + 2 añadidos

    Ejecución original
    -------------------
    -> ./scripts/run.sh -p NeurolAgent

    Ejecución nuevo agente (4 factores heurísitcos)
    -------------------
    -> ./scripts/run.sh -p NeurolAgent2
    """
    @override
    def traditionalEvaluation(self, state):
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()

        # Mejorar evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()

        #-----------------#
        # Factores (4)
        #-----------------#

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

        # Factor 3: Huida proporcional a los fantasmas
        """
        Cuanto más cerca estén los fantasmas, más se penaliza inversamente
        proporcional a la distancia de cualquier fanstasma sin asustar
        Si pacman sobrevive más tiempo huyendo de los fantasmas más puntos,
        tiende a evitar situaciones donde se quede encerrado
        """
        for ghost_state in ghost_states:
            if ghost_state.scaredTimer == 0:
                ghost_posicion = ghost_state.getPosition()
                ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
                score -= 150 / (ghost_distance + 0.5)

        # Factor 4: Concentración de comida (cluster)
        """
        Penalizamos cuando hay comida dispersa por el mapa. Se calcula como la
        distancia media a los 5 cocos más cercanos. Si el valor es alto, hay dispersión
        luego pacman tiene que recorrer mucho. Si el valor es bajo: cluster -> pacman
        puede "limpiar" esa zona y terminar la partida antes. Gracias a este factor
        pacman puede terminar partidas más rápidamente
        """
        if food:
            distances_to_food = sorted(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            cercana = distances_to_food[:5]
            media_cluster_dist = sum(cercana) / len(cercana)
            score -= 0.4 * media_cluster_dist

        # Factor 5: Casos Limite
        """
        Simplemente le otorgamos valores infinitos o infinitos negativos
        a situaciones de victoria segura o de derrota segura respectivamente,
        asegurando asi que no se suicida o pierde la oportunidad de ganar.
        """
        if state.isLose(): return float('-inf')
        if state.isWin(): return float('-inf')

        return score

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

###########################################################################
# Stefano
###########################################################################

class AlphaBetaNeuralAgent(NeuralAgent, AlphaBetaAgent):
    ''''''

    # Weight of the neural score
    nweight: float = float(os.getenv('NEURAL_WEIGHT', 0.9))

    @override
    def evaluationFunction(self, state) -> float:
        """
        Una función de evaluación basada en la red neuronal y en alphabeta con heurísticas.
        """
        return self.traditionalEvaluation(state) *  (1 - type(self).nweight) + self.neuralEvaluation(state) * type(self).nweight

    @override
    def getAction(self, gameState: 'GameState') -> str:
        '''
        Calcula la mejor acción combinando un prior de política neuronal (Policy Network)
        en el nodo raíz con una búsqueda adversaria (Alpha-Beta) para evaluar las consecuencias.

        :param gameState: Estado actual de la partida.
        :return: String con la acción elegida (ej. 'North', 'Stop').
        '''
        self.move_count += 1

        if self.model is None:
            print('ERROR: Modelo no cargado. Saliendo...')
            exit()

        legal_actions = gameState.getLegalActions(0)

        # 1. Policy Prior: Evaluación directa con la red neuronal del estado actual
        state_matrix = self.state_to_matrix(gameState)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        # Extraer probabilidades solo para acciones legales
        action_probs: dict[str, float] = {}
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs[action] = float(prob)

        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # 2. Búsqueda Alpha-Beta + Integración del Prior Neuronal
        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)

            # El valor devuelto ya está ponderado internamente por evaluationFunction en los nodos hoja
            ab_score = self._alphaBeta(1, 0, successor, alpha, beta)

            # Prior de la red para la acción raíz, escalado y ponderado por nweight
            neural_action_score = (action_probs[action] * 100.0) * type(self).nweight

            # Combinación: Búsqueda (Value) + Instinto (Policy)
            combined_score = ab_score + neural_action_score

            # Penalización heurística determinista
            if action == 'Stop' and len(legal_actions) > 1:
                combined_score -= 50.0

            # Actualización del mejor nodo
            if combined_score > best_score:
                best_score = combined_score
                best_action = action

            # Actualización de Alpha para la poda en la raíz
            alpha = max(alpha, best_score)

        return best_action

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)