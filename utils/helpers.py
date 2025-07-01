import random
from typing import List, Tuple, Set
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from tqdm import tqdm
import torch


class Node:
  def __init__(self, x:int, y:int):
    self.coords = (x,y)
    self.edges = [True,True,True,True] # L R T B
    self.flags = None # 0 - Start Node ; 1 - End Node

  def _get_direction(self, coords_1 : Tuple, coords_2 : Tuple):

    dir_dict = {
        'Left' : 0,
        'Right' : 1,
        'Top' : 2,
        'Bottom' : 3,
    }

    # Check for right
    if coords_2[0] == coords_1[0]:
      if coords_2[1] > coords_1[1]:
        return dir_dict['Right'],dir_dict['Left']
      else:
        return dir_dict['Left'],dir_dict['Right']

    else:
      if coords_2[0] > coords_1[0]:
        return dir_dict['Bottom'],dir_dict['Top']
      else:
        return dir_dict['Top'],dir_dict['Bottom']

    return None


  def connect_adj_nodes(self, node_2 : 'Node'):
    erase_wall_1,erase_wall_2 = self._get_direction(self.coords,node_2.coords)
    self.edges[erase_wall_1] = False
    node_2.edges[erase_wall_2] = False

class Graph:
  def __init__(self, size : int = 28):
    self.size = size
    self.num_nodes = size * size # Size = 6 => Total nodes in the one side extreme position

  def create_graph(self):
    return [[Node(row,col)for col in range(self.size)]for row in range(self.size)]


# ----------------------Dataset Generation Algorithim--------------------------------
class DFS:
    def __init__(self, graph: List[List['Node']]):
        self._graph = graph

    def _get_adjacent_nodes(self, node_cell: 'Node', size: int) -> List['Node']:
        row, col = node_cell.coords
        adj_nodes = []

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < size and 0 <= new_col < size:
                adj_nodes.append(self._graph[new_row][new_col])

        return adj_nodes

    def _random_start_selector(self) -> 'Node':
        size = len(self._graph)
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        start_node = self._graph[row][col]
        start_node.flag = 0  # Start Node (optional)
        return start_node

    def create_spt(self):
        visited: Set['Node'] = set()
        stack: List['Node'] = []

        start_node = self._random_start_selector()
        visited.add(start_node)
        stack.append(start_node)

        while stack:
            curr_node = stack[-1]
            neighbors = self._get_adjacent_nodes(curr_node, size=len(self._graph))
            unvisited_neighbors = [n for n in neighbors if n not in visited]

            if unvisited_neighbors:
                next_node = random.choice(unvisited_neighbors)
                curr_node.connect_adj_nodes(next_node)
                visited.add(next_node)
                stack.append(next_node)
            else:
                stack.pop()

        return self._graph

class RandomizedPrims:
    def __init__(self, graph: List[List['Node']]):
        self._graph = graph
        self.size = len(graph)

    def _random_start_selector(self) -> 'Node':
        row = random.randint(0, self.size - 1)
        col = random.randint(0, self.size - 1)
        return self._graph[row][col]

    def _get_adjacent_nodes(self, node: 'Node') -> List['Node']:
        row, col = node.coords
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, Bottom, Left, Right

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append(self._graph[nr][nc])

        return neighbors

    def create_spt(self):
        visited: Set['Node'] = set()
        frontier: List[Tuple['Node', 'Node']] = []

        start_node = self._random_start_selector()
        visited.add(start_node)

        for neighbor in self._get_adjacent_nodes(start_node):
            frontier.append((start_node, neighbor))

        while frontier:
            # Choose a random edge from the frontier
            curr_node, next_node = random.choice(frontier)
            frontier.remove((curr_node, next_node))

            if next_node not in visited:
                curr_node.connect_adj_nodes(next_node)
                visited.add(next_node)

                for neighbor in self._get_adjacent_nodes(next_node):
                    if neighbor not in visited:
                        frontier.append((next_node, neighbor))

        return self._graph

class DisjointSet:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])  # Path compression
        return self.parent[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.parent[root2] = root1
            return True
        return False  # Already connected


class Kruskals:
  def __init__(self, graph):
    self.graph = graph
    self.size = len(graph)

  def _get_adjacent_nodes(self, node: 'Node') -> List['Node']:
      row, col = node.coords
      neighbors = []
      directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, Bottom, Left, Right

      for dr, dc in directions:
          nr, nc = row + dr, col + dc
          if 0 <= nr < self.size and 0 <= nc < self.size:
              neighbors.append(self.graph[nr][nc])
      return neighbors

  def _get_all_edges(self):
      all_nodes = sum(self.graph, [])  # Flatten 2D list
      all_edges = set()

      for node in all_nodes:
          adj_nodes = self._get_adjacent_nodes(node)
          for adj in adj_nodes:
              # Normalize the edge so (A, B) and (B, A) are treated the same
              edge = tuple(sorted((node, adj), key=lambda n: n.coords))
              all_edges.add(edge)  # set will automatically deduplicate

      return list(all_edges)

  def create_spt(self):
      all_edges = self._get_all_edges()
      random.shuffle(all_edges)

      all_nodes = sum(self.graph, [])
      ds = DisjointSet(all_nodes)

      for node1, node2 in all_edges:
          if ds.union(node1, node2):  # They were not connected
              node1.connect_adj_nodes(node2)

      return self.graph



class HuntAndKill:
    def __init__(self, graph):
        self.graph = graph
        self.size = len(graph)

    def _select_random_cell(self):
        return random.choice(sum(self.graph, []))

    def _get_adjacent_nodes(self, node: 'Node') -> List['Node']:
        row, col = node.coords
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Top, Bottom, Left, Right

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append(self.graph[nr][nc])

        return neighbors

    def _do_random_walk(self, start_node: 'Node', visited: Set['Node']) -> 'Node':
        current = start_node

        while True:
            visited.add(current)
            unvisited_neighbors = [
                n for n in self._get_adjacent_nodes(current) if n not in visited
            ]

            if not unvisited_neighbors:
                return None  # End of this walk

            next_node = random.choice(unvisited_neighbors)
            current.connect_adj_nodes(next_node)
            current = next_node

    def _activate_hunt_phase(self, visited: Set['Node']) -> 'Node':
        for row in self.graph:
            for node in row:
                if node not in visited:
                    neighbors = self._get_adjacent_nodes(node)
                    visited_neighbors = [n for n in neighbors if n in visited]
                    if visited_neighbors:
                        chosen = random.choice(visited_neighbors)
                        node.connect_adj_nodes(chosen)
                        return node  # Start new kill phase from here
        return None  # No unvisited node with visited neighbor found

    def create_spt(self):
        visited = set()
        current_node = self._select_random_cell()

        while current_node:
            self._do_random_walk(current_node, visited)
            current_node = self._activate_hunt_phase(visited)

        return self.graph


class Wilson:
    def __init__(self, graph: 'Graph'):
        self.graph = graph
        self.size = len(graph)

    def _select_random_cell(self):
        return random.choice(random.choice(self.graph))

    def _get_adjacent_nodes(self, node: 'Node') -> List['Node']:
        row, col = node.coords
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append(self.graph[nr][nc])
        return neighbors

    def _do_loop_erase(self, path_list, to_node):
        # Keep everything up to and including to_node
        return path_list[0:path_list.index(to_node) + 1]

    def _do_random_walk(self, from_node, visited):
        path_list = [from_node]
        current_node = from_node  # Track the current position properly

        while True:
            neighbours = self._get_adjacent_nodes(current_node)
            to_node = random.choice(neighbours)

            if to_node in path_list:
                # Loop detected → erase
                path_list = self._do_loop_erase(path_list, to_node)
                current_node = to_node  # Update current position to the loop point

            elif to_node in visited:
                # End of walk reached (connected to existing tree)
                path_list.append(to_node)
                return path_list

            else:
                # Normal case → keep walking
                path_list.append(to_node)
                current_node = to_node  # Update current position

    def _remove_walls(self, path_list):
        for node1, node2 in zip(path_list, path_list[1:]):
            node1.connect_adj_nodes(node2)

    def create_spt(self):
        visited = set()

        starting_node = self._select_random_cell()
        visited.add(starting_node)

        while len(visited) < self.size ** 2:
            rndm_node = self._select_random_cell()
            if rndm_node not in visited:
                path_list = self._do_random_walk(rndm_node, visited)
                if path_list:  # Safety check
                    self._remove_walls(path_list)
                    visited.update(path_list)

        return self.graph


class DFS_Solver:
    def __init__(self, spt, start_node, end_node):
        self.spt = spt  # 2D grid of nodes
        self.start_node = self._get_node_at(*start_node)
        self.end_node = self._get_node_at(*end_node)

    def _get_node_at(self, x, y):
        for node in sum(self.spt, []):
            if node.coords == (x, y):
                return node
        return None

    def _get_connected_neighbors(self, node):
        edges = node.edges  # [Left, Right, Top, Bottom]
        x, y = node.coords
        neighbors = []

        if edges[0] == 0:  # Left
            neighbors.append(self._get_node_at(x, y - 1))
        if edges[1] == 0:  # Right
            neighbors.append(self._get_node_at(x, y + 1))
        if edges[2] == 0:  # Top
            neighbors.append(self._get_node_at(x - 1, y))
        if edges[3] == 0:  # Bottom
            neighbors.append(self._get_node_at(x + 1, y))

        return [n for n in neighbors if n is not None]

    def _dfs(self, current, visited):
        if current == self.end_node:
            return [current]

        visited.add(current)

        for neighbor in self._get_connected_neighbors(current):
            if neighbor not in visited:
                path = self._dfs(neighbor, visited)
                if path:
                    return [current] + path

        return None

    def create_path(self):
        visited = set()
        return self._dfs(self.start_node, visited)
    

from PIL import Image

class Visualizer:
    def __init__(self):
        pass

    def maze_visualizer(self, graph, cell_size=1, wall_size=1, start_coords=None, end_coords=None):
        size = len(graph)
        pixel_per_cell = cell_size + wall_size
        img_size = size * pixel_per_cell + wall_size

        # Create a black image (walls)
        img = Image.new("RGB", (img_size, img_size), "black")
        pixels = img.load()

        for row in range(size):
            for col in range(size):
                node = graph[row][col]
                x = col * pixel_per_cell + wall_size
                y = row * pixel_per_cell + wall_size

                # Determine cell color
                if (row, col) == start_coords or (row, col) == end_coords:
                    color = (0, 255, 0)  # Green for start or end
                else:
                    color = (255, 255, 255)  # White for path

                # Fill the cell
                for dy in range(cell_size):
                    for dx in range(cell_size):
                        px, py = x + dx, y + dy
                        if 0 < px < img_size - 1 and 0 < py < img_size - 1:
                            pixels[px, py] = color

                # Draw connections (open walls)
                directions = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]  # L, R, T, B
                for dr, dc, wall_index in directions:
                    if not node.edges[wall_index]:
                        for w in range(wall_size):
                            if dr == 0:  # Horizontal
                                px = x + dc * wall_size
                                py = y + w
                            else:  # Vertical
                                px = x + w
                                py = y + dr * wall_size
                            if 0 < px < img_size - 1 and 0 < py < img_size - 1:
                                pixels[px, py] = (255, 255, 255)

        return img
    
    def create_solution_image(self, spt, path, cell_size=1, wall_size=1,
                          start_coords=None, end_coords=None,
                          path_color=(255, 0, 0)):

      img = self.maze_visualizer(spt, cell_size=cell_size, wall_size=wall_size,
                                start_coords=start_coords, end_coords=end_coords)
      pixels = img.load()
      pixel_per_cell = cell_size + wall_size

      for i in range(len(path)):
          row, col = path[i].coords
          x = col * pixel_per_cell + wall_size
          y = row * pixel_per_cell + wall_size

          # Color the cell
          for dy in range(cell_size):
              for dx in range(cell_size):
                  px, py = x + dx, y + dy
                  if 0 <= px < img.width and 0 <= py < img.height:
                      pixels[px, py] = path_color

          # If there's a next node, draw the passage to it
          if i < len(path) - 1:
              next_row, next_col = path[i + 1].coords
              dx = next_col - col
              dy = next_row - row

              if dx != 0 or dy != 0:
                  wall_x = x + (dx * (cell_size + wall_size)) // 2
                  wall_y = y + (dy * (cell_size + wall_size)) // 2

                  for w in range(wall_size):
                      if dx != 0:  # Horizontal move
                          px = wall_x
                          py = y + w
                      else:        # Vertical move
                          px = x + w
                          py = wall_y

                      if 0 <= px < img.width and 0 <= py < img.height:
                          pixels[px, py] = path_color

      return img





# --------------------------------------------- MODEL RELATED ----------------------------------------------------



def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['test_acc'], label='Test Accuracy', color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Metrics
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:\n")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()