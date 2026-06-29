import pygame


pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Mi primera ventana con Pygame")

cell_size = 80

mini_maze = [
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
]

goal_pos = (1, 2)
start_pos = (1, 1)
agent_pos = start_pos

controls = {
    pygame.K_w: (-1, 0),
    pygame.K_s: (1, 0),
    pygame.K_a: (0, -1),
    pygame.K_d: (0, 1),
}

def is_valid_position(row, col):
    if row < 0 or row >= len(mini_maze):
        return False

    if col < 0 or col >= len(mini_maze[row]):
        return False

    if mini_maze[row][col] == 1:
        return False

    return True

running = True
done = False


while running:
    screen.fill((200, 200, 200))
    
    for row in range(len(mini_maze)):
        for col in range(len(mini_maze[row])):
            x = col * cell_size
            y = row * cell_size

            cell_rect = pygame.Rect(x, y, cell_size, cell_size)

            pos = (row, col)

            if mini_maze[row][col] == 1:
                pygame.draw.rect(screen, (40, 40, 40), cell_rect)
            elif pos == goal_pos:
                pygame.draw.rect(screen, (80, 200, 120), cell_rect)
            else:
                pygame.draw.rect(screen, (235, 235, 235), cell_rect)
            
            pygame.draw.rect(screen, (180, 180, 180), cell_rect, 1)

    agent_row, agent_col = agent_pos

    center_x = agent_col * cell_size + cell_size // 2
    center_y = agent_row * cell_size + cell_size // 2

    pygame.draw.circle(screen, (70, 130, 255), (center_x, center_y), cell_size // 3)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            print("Tecla presionada:", event.key)

            if event.key == pygame.K_r:
                agent_pos = start_pos
                done = False
                print("Episodio reiniciado.")

            if event.key in controls and not done:
                move_row, move_col = controls[event.key]

                agent_row, agent_col = agent_pos

                new_row = agent_row + move_row
                new_col = agent_col + move_col

                if is_valid_position(new_row, new_col):
                    agent_pos = (new_row, new_col)

                    if agent_pos == goal_pos:
                        print("¡El agente llegó a la meta!")
                        done = True
    
    pygame.display.flip()

pygame.quit()