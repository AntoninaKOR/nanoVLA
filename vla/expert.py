"""
Expert policy for MiniGrid EmptyEnv using BFS shortest-path planning.

The EmptyEnv has a single goal in the bottom-right corner. The agent starts
at a random position/orientation. We use BFS on the grid to find the shortest
path, then convert it into a sequence of (turn_left, turn_right, move_forward)
actions considering the agent's current orientation.
"""
from collections import deque

from minigrid.core.constants import DIR_TO_VEC


# Direction indices in MiniGrid: 0=right, 1=down, 2=left, 3=up
DIR_VECTORS = {i: tuple(DIR_TO_VEC[i]) for i in range(4)}
VEC_TO_DIR = {v: k for k, v in DIR_VECTORS.items()}


def bfs_path(grid, start, goal, width, height):
    """BFS to find shortest path on the grid, avoiding walls."""
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                cell = grid.get(nx, ny)
                if cell is None or cell.type == "goal":
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return None  # No path found


def path_to_actions(path, start_dir):
    """
    Convert a grid path into a sequence of action names given a starting direction.

    Returns list of action names: 'turn_left', 'turn_right', 'move_forward'
    """
    actions = []
    current_dir = start_dir

    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx, ny = path[i + 1]
        desired_vec = (nx - cx, ny - cy)
        desired_dir = VEC_TO_DIR[desired_vec]

        # Turn to face the desired direction
        while current_dir != desired_dir:
            # Determine shortest turn
            diff = (desired_dir - current_dir) % 4
            if diff == 1:
                actions.append("turn_right")
                current_dir = (current_dir + 1) % 4
            elif diff == 3:
                actions.append("turn_left")
                current_dir = (current_dir - 1) % 4
            elif diff == 2:
                # 180 degrees - two right turns
                actions.append("turn_right")
                current_dir = (current_dir + 1) % 4
                actions.append("turn_right")
                current_dir = (current_dir + 1) % 4

        actions.append("move_forward")

    return actions


def get_expert_actions(env):
    """
    Get the full sequence of expert actions for the current environment state.

    Returns:
        list of action name strings, or None if no path exists.
    """
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    goal_pos = None

    grid = env.unwrapped.grid
    width = grid.width
    height = grid.height

    # Find goal position
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                goal_pos = (x, y)
                break
        if goal_pos:
            break

    if goal_pos is None:
        return None

    path = bfs_path(grid, agent_pos, goal_pos, width, height)
    if path is None:
        return None

    return path_to_actions(path, agent_dir)
