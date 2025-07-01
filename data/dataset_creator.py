from MazeGen.utils.helpers import (
    Graph, DFS, RandomizedPrims, Wilson,
    HuntAndKill, Kruskals, DFS_Solver, Visualizer
)
from typing import List
from pathlib import Path
import random
from PIL import ImageOps

def create_dataset(algorithms: List[str], n_explore: int = 5000, graph_size: int = 10):
    print("ENTRY")

    def _get_maze_generator(name):
        name = name.lower()  # Makes it case-insensitive
        if name == 'dfs':
            return DFS
        elif name == 'wilson':
            return Wilson
        elif name == 'hak':
            return HuntAndKill
        elif name == 'r_prims':
            return RandomizedPrims
        elif name == 'r_kruskals':
            return Kruskals
        else:
            raise ValueError(f'Invalid Argument Passed: {name}')

    def _get_border_nodes(graph_size):
        border_nodes = []
        for col in range(graph_size):
            border_nodes.append((0, col))  # Top
            border_nodes.append((graph_size - 1, col))  # Bottom
        for row in range(1, graph_size - 1):
            border_nodes.append((row, 0))  # Left
            border_nodes.append((row, graph_size - 1))  # Right
        return border_nodes

    # Root directory for the dataset
    root_dir = Path('MazeGen') / 'data' / 'ImageDataset'
    graph = Graph(graph_size)
    viz = Visualizer()
    border_coords = _get_border_nodes(graph_size)

    for alg_name in algorithms:
        alg_dir = root_dir / alg_name
        maze_path = alg_dir / 'mazes'
        solution_path = alg_dir / 'solutions'
        maze_path.mkdir(parents=True, exist_ok=True)
        solution_path.mkdir(parents=True, exist_ok=True)

        print(f'Creating dataset for {alg_name}...')

        for i in range(n_explore):
            try:
                grid = graph.create_graph()
                MazeGenerator = _get_maze_generator(alg_name)
                spt = MazeGenerator(grid).create_spt()

                start_coord = random.choice(border_coords)
                end_coord = random.choice(border_coords)
                while end_coord == start_coord:
                    end_coord = random.choice(border_coords)

                start_node = spt[start_coord[0]][start_coord[1]]
                end_node = spt[end_coord[0]][end_coord[1]]

                maze_img = viz.maze_visualizer(
                    spt,
                    start_coords=start_node.coords,
                    end_coords=end_node.coords
                )
                maze_img = ImageOps.expand(maze_img, border=(4, 4, 3, 3), fill=0)
                maze_img.save(maze_path / f"{i:05d}.png")

                # ✅ FIXED: pass coords instead of Node
                solver = DFS_Solver(spt, start_node.coords, end_node.coords)
                path = solver.create_path()

                if path:
                    solution_img = viz.create_solution_image(
                        spt,
                        path=path,
                        start_coords=start_node.coords,
                        end_coords=end_node.coords
                    )
                    solution_img = ImageOps.expand(solution_img, border=(4, 4, 3, 3), fill=0)
                    solution_img.save(solution_path / f"{i:05d}.png")
                else:
                    print(f"No path found for {alg_name} maze {i}")

                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{n_explore} mazes for {alg_name}")

            except Exception as e:
                print(f"Error generating maze {i} for {alg_name}: {e}")
                continue

        print(f'Completed dataset creation for {alg_name}')
