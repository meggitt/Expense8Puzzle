from queue import LifoQueue
import sys
import datetime
from queue import PriorityQueue, Queue, LifoQueue
import heapq


class PuzzleTile:
    goalState = []
    h = None
    evalFunc = None
    isHeuristicRequired = False
    total = 0

    def __init__(self, state, parent, action, cost, heuristicRequired=False, depth=0):
        self.parent = parent
        self.state = state
        self.depth = depth
        self.action = action
        if parent:
            self.cost = parent.cost + cost
        else:
            self.cost = cost
        if heuristicRequired:
            self.isHeuristicRequired = True
            self.generateHeuristic()
            self.evalFunc = self.h+self.cost
        PuzzleTile.total += 1

    def generateHeuristic(self):
        self.h = 0
        state = []
        goal_state = []
        for i in range(3):
            state += self.state[i]
            goal_state += self.goalState[i]
        for n in range(1, 9):
            distance = abs(state.index(n) - goal_state.index(n))
            i = int(distance/3)
            j = int(distance % 3)
            self.h = self.h+i+j

    def generateHeuristicGreedy(self, ):
        self.generateHeuristic()
        return self.h+self.cost

    def __str__(self):
        return f"<  state = {self.state} ,action = {{{self.action}}}, g(n) = {self.cost}, d = {self.depth} , f(n) = {self.cost}, Parent = Pointer to {self.state} >"

    def isGoalState(self):
        for i in range(3):
            for j in range(3):
                if self.state[i] != self.goalState[i]:
                    return False
        return True

    @staticmethod
    def moves(i, j):
        choices = ['Left', 'Up', 'Right', 'Down']
        if i == 0:
            choices.remove('Down')
        elif i == 2:
            choices.remove('Up')
        if j == 0:
            choices.remove('Right')
        elif j == 2:
            choices.remove('Left')
        return choices

    def generateSuccessorNodes(self, f):
        children = []
        x, y = next((i, j) for i in range(3)
                    for j in range(3) if self.state[i][j] == 0)
        available_paths = self.moves(x, y)

        if f is not None:
            f.write("\nFringe:")

        for direction in available_paths:
            dx, dy = {'Down': (-1, 0), 'Up': (1, 0),
                      'Left': (0, 1), 'Right': (0, -1)}[direction]
            o = self.state[x + dx][y + dy]
            new_state = [row[:] for row in self.state]
            new_state[x][y], new_state[x + dx][y +
                                               dy] = new_state[x + dx][y + dy], new_state[x][y]
            new_node = PuzzleTile(
                new_state, self, f"\tMove {o} {direction}", 1, self.isHeuristicRequired, self.depth + 1)
            if f is not None:
                f.write("\n" + new_node.__str__())
            children.append(new_node)
        return children

    def gofn(self):
        totalCosts = self.cost
        parent = self.parent
        while parent:
            totalCosts += parent.cost
            parent = parent.parent
        return totalCosts

    def finalMove(self):
        solutionNodeList = []
        solutionNodeList.append(self.__str__())
        move = [self.action]
        path = self
        cost = int(path.action.split()[1])
        while path.parent is not None:
            path = path.parent
            if path.action is not None:
                cost += int(path.action.split()[1])
                move.append(path.action)
            solutionNodeList.append(path.__str__())
        solution_node_list = solutionNodeList[::-1]
        move = move[::-1]
        return [solution_node_list[1:], cost, move]


def printSolution(x, node, nodes_popped, nodes_expanded, max_fringe):
    ans = node.finalMove()
    info = [f"Nodes Popped = {nodes_popped}",
            f"Nodes Expanded = {nodes_expanded}",
            f"Nodes Generated = {PuzzleTile.total}",
            f"Max Fringe Size= {max_fringe}",
            f"Solution found at depth {len(ans[0])} with cost of {ans[1]}",
            "Steps:"]
    print('\n'.join(info[0:]))
    if (x != None):
        x.writelines(info)
    return ans[2]


def BFS(initial_state, f):
    start_node = PuzzleTile(initial_state, None, None, 0)

    if start_node.isGoalState():
        return start_node.find_result_action(f)

    queue = Queue()
    queue.put(start_node)
    explored_nodes = []
    total_expanded_nodes = 0
    total_popped_nodes = 0
    max_fringe_size = 0

    while not queue.empty():
        node = queue.get()
        explored_nodes.append(node.state)
        total_expanded_nodes += 1

        children = node.generateSuccessorNodes(f)

        if f is not None:
            f.write("\n{} successors generated".format(len(children)))
            f.write("\nClosed: " + str(explored_nodes))

        max_fringe_size = max(max_fringe_size, queue.qsize())

        for child in children:
            if child.state not in explored_nodes:
                queue.put(child)
                if child.isGoalState():
                    return printSolution(f, child, total_popped_nodes, total_expanded_nodes, max_fringe_size)
            else:
                total_popped_nodes += 1

    return


def UCS(initial_state, x=None):
    def search(initial_state, heuristic_fn, x):
        priority_queue = []
        cost = 0
        initial_node = PuzzleTile(initial_state, None, None, 0)
        if x is not None:
            x.write(f"\nGenerating successors to {initial_node.__str__()}")
        total_expanded_nodes = 0
        max_fringe_size = -1
        while not initial_node.isGoalState():
            children = initial_node.generateSuccessorNodes(x)
            total_expanded_nodes += 1
            if x is not None:
                x.write(f"\n{len(children)} successors generated")
            for child in children:
                item = (heuristic_fn(child), cost, child)
                heapq.heappush(priority_queue, item)
                cost += 1
            initial_node = heapq.heappop(priority_queue)[2]
            max_fringe_size = max(max_fringe_size, len(priority_queue))

        return printSolution(x, initial_node, cost, total_expanded_nodes, max_fringe_size)

    def heuristic_fn(node):
        return node.gofn()
    return search(initial_state, heuristic_fn, x)


def DFS(initial_state, x):
    root = PuzzleTile(initial_state, None, None, 0)
    explored_nodes = []
    total_popped_nodes = 0
    total_expanded_nodes = 0
    maxFringeSize = 0
    if root.isGoalState():
        return printSolution(x, root, total_popped_nodes, total_expanded_nodes, maxFringeSize)
    queue = LifoQueue()
    queue.put(root)
    while not (queue.empty()):
        currentNode = queue.get()
        explored_nodes.append(currentNode.state)
        total_expanded_nodes += 1
        maxFringeSize = max(maxFringeSize, len(queue.queue))
        children = currentNode.generateSuccessorNodes(x)
        if (x != None):
            x.write("\n{} successors generated".format(len(children)))
        for child in children:
            if child.state not in explored_nodes:
                if child.isGoalState():
                    return printSolution(x, child, total_popped_nodes, total_expanded_nodes, maxFringeSize)
                queue.put(child)
            else:
                total_popped_nodes += 1


def DLS(initial_state, f):
    depth_limit = int(input("Enter allowed depth:"))
    node = PuzzleTile(initial_state, None, None, 0)
    total_popped_nodes = 0
    total_expanded_nodes = 0
    max_fringe_size = 0

    if node.isGoalState():
        return printSolution(f, node, total_popped_nodes, total_expanded_nodes, max_fringe_size)

    queue = LifoQueue()
    queue.put(node)
    explored_nodes = []

    while not queue.empty():
        current_node = queue.get()
        max_depth = current_node.depth
        explored_nodes.append(current_node.state)
        total_expanded_nodes += 1
        max_fringe_size = max(max_fringe_size, queue.qsize())

        if max_depth == depth_limit:
            continue

        children = current_node.generateSuccessorNodes(f)

        if f is not None:
            f.write("\n{} successors generated".format(len(children)))

        for child in children:
            if child.state not in explored_nodes:
                if child.isGoalState():
                    return printSolution(f, child, total_popped_nodes, total_expanded_nodes, max_fringe_size)
                queue.put(child)
            else:
                total_popped_nodes += 1

    print("Could not find a solution using this algorithm.")
    return


def IDS(initial_state, x):
    total_expanded_nodes = 0
    total_popped_nodes = 0

    def dls(node, depth):
        nonlocal total_expanded_nodes, total_popped_nodes
        total_expanded_nodes += 1
        if depth == 0:
            return None
        if node.isGoalState():
            return node
        children = node.generateSuccessorNodes(x)
        if x is not None:
            x.write("\n{} successors generated".format(len(children)))
        for child in children:
            total_popped_nodes += 1
            result = dls(child, depth - 1)
            if result:
                return result
        return None

    depth = 0
    while True:
        answer = dls(PuzzleTile(initial_state, None, None, 0), depth)
        if answer:
            return printSolution(x, answer, total_popped_nodes, total_expanded_nodes, depth)
        depth += 1


def Greedy(initial_state, x):
    queue = PriorityQueue()
    explored_nodes = []
    counter = 0
    total_popped_nodes = 0
    total_expanded_nodes = 0
    start_node = PuzzleTile(initial_state, None, None, 0, True)
    queue.put((start_node.generateHeuristicGreedy(), counter, start_node))
    max_fringe_size = -1

    while not queue.empty():
        current_node = queue.get()[2]
        total_expanded_nodes += 1
        explored_nodes.append(current_node.state)
        max_fringe_size = max(len(queue.queue), max_fringe_size)

        if current_node.isGoalState():
            return printSolution(x, current_node, total_popped_nodes, total_expanded_nodes, max_fringe_size)

        children = current_node.generateSuccessorNodes(x)

        if x is not None:
            x.write("\n{} successors generated".format(len(children)))
            x.write("\nClosed: " + str(explored_nodes))

        for child in children:
            if child.state not in explored_nodes:
                counter += 1
                queue.put((child.generateHeuristicGreedy(), counter, child))
            else:
                total_popped_nodes += 1
    return


def AStar(initial_state, x=None):
    count = 0
    explored_nodes = []
    start_node = PuzzleTile(initial_state, None, None, 0, True)
    fringe = PriorityQueue()
    fringe.put((start_node.evalFunc, count, start_node))
    popped_nodes_count = 0
    max_fringe_size = 0
    while not fringe.empty():
        max_fringe_size = max(len(fringe.queue), max_fringe_size)
        current_node = fringe.get()[2]
        if x is not None:
            x.write("\nGenerating successors to {}".format(
                current_node.__str__()))
        explored_nodes.append(current_node.state)
        if current_node.isGoalState():
            return printSolution(x, current_node, count, popped_nodes_count, max_fringe_size)
        children = current_node.generateSuccessorNodes(x)
        if x is not None:
            x.write("\n{} successors generated".format(len(children)))
            x.write("\nClosed: " + str(explored_nodes))
        for child in children:
            if child.state not in explored_nodes:
                count += 1
                fringe.put((child.evalFunc, count, child))
            else:
                popped_nodes_count += 1
        if x is not None:
            x.write("\nClosed: " + str(explored_nodes))

    return


if __name__ == "__main__":
    sys.setrecursionlimit(10**9)
    sysargs = sys.argv
    startFile = open(sysargs[1], 'r')
    goalFile = open(sysargs[2], 'r')
    start = [list(map(int, line[:-1].split()))
             for line in startFile.readlines()[:-1]]
    goal = [list(map(int, line[:-1].split()))
            for line in goalFile.readlines()[:-1]]
    dumpFile = None
    if (len(sysargs) < 4):
        method = "a*"
        isDumpFileRequired = False
    if (len(sysargs) == 4 and str(sysargs[3]).lower() != 'true' and str(sysargs[3]).lower() != 'false'):
        method = "a*"
        isDumpFileRequired = False
    if (len(sysargs) == 4 and (str(sysargs[3]).lower() == 'true' or str(sysargs[3]).lower() == 'false')):
        method = "a*"
        isDumpFileRequired = True if (
            str(sysargs[3]).lower() == 'true') else False
    if (len(sysargs) == 5):
        method = str(sysargs[3]).lower()
        isDumpFileRequired = True if (
            str(sysargs[4]).lower() == 'true') else False
    if isDumpFileRequired:
        filename = datetime.datetime.now().__str__().replace(":", "-")
        dumpFile = open("trace{}.txt".format(filename), 'w')
        dumpFile.writelines(["Command-Line Arguments : ['{}', '{}', '{}', '{}']".format(startFile.name, goalFile.name,
                            method, isDumpFileRequired), "\nMethod Selected: {}".format(method), "\nRunning {}".format(method),])
    mappedAlgorithms = {'bfs': BFS, 'ucs': UCS, 'dfs': DFS,
                        'dls': DLS, 'ids': IDS, 'greedy': Greedy, 'a*': AStar, }
    PuzzleTile.goalState = goal
    steps = mappedAlgorithms[method](start, dumpFile)
    if (steps != None):
        print('\n'.join(steps))
