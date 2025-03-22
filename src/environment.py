import random
from collections import namedtuple
from enum import Enum
from memory import Episode_Memory as Memory
import matplotlib as plt
import numpy as np
import pygame

# Initialize Pygame and the font for rendering text
pygame.init()  # Initialize all Pygame modules
font = pygame.font.Font("arial.ttf", 25)  # Load font for rendering text

# Define directions as an Enum for clarity
class Direction(Enum):
    RIGHT = 0  # Move to the right
    DOWN = 1   # Move to the left
    LEFT = 2   # Move downward
    UP = 3   # Move upward

# Create a named tuple to represent points (x, y) on the grid
Point = namedtuple("Point", "x, y")

# Define RGB color values
WHITE = (255, 255, 255)  # Color for text
RED = (200, 0, 0)        # Color for the window
BLUE1 = (0, 0, 255)      # Color for the outer Quad
BLUE2 = (0, 100, 255)    # Color for the inner Quad
BLACK = (0, 0, 0)        # Background color

# Set the block size and game speed
BLOCK_SIZE = 20  # Size of each block
SPEED = 10    # Game speed in frames per second

# Define the Environment class
class Environment:
    def __init__(self, w=300, h=260, render=False):
        self.render = render  # Set the rendering flag
        self.done = False
        self.memory = Memory()
        self.option = 0
        self.w = w
        self.h = h
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Quad")
        self.clock = pygame.time.Clock()
        self.reset()
    def reset(self):
        """
        Reset the game state for a new game.
        """
        # directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # self.direction = random.choice(directions)  # Start with the direction random


        self.direction = Direction.RIGHT
        self.Quad = Point(20, 20)  # Start the Quad at the initial position


        self.score = 0  # Initialize the score to zero
        self.window = None  # Initialize the window to None
        self.frame_iteration = 0  # Initialize the frame iteration counter

        # Place the window before getting the state
        self.place_window()  # Call place_window to randomly place the window on the screen

        # Now you can safely get the current state of the game
        current_state = self.state()  # Get the current state of the game

        return current_state  # Return the initial state

    def place_window(self):
        """
        Place a window (gap) randomly on the screen.
        The window is either vertical or horizontal.
        """

        # Randomly choose between two options (horizontal or vertical)
        self.option = random.choice(["1"])

        # If option 1 is selected, create a horizontal window
        if self.option == "0":
            # Randomly generate x and y coordinates for the first side of the window
            x = random.randint(5, ((self.w - BLOCK_SIZE) // BLOCK_SIZE) - 5) * BLOCK_SIZE
            y = random.randint(5, ((self.h - BLOCK_SIZE) // BLOCK_SIZE) - 5) * BLOCK_SIZE

            # Define the first and second sides of the window (horizontal)
            self.first_side = Point(x, y)
            self.second_side = Point(x + (6 * BLOCK_SIZE), y)

        # If option 2 is selected, create a vertical window
        elif self.option == "1":
            # Randomly generate x and y coordinates for the first side of the window
            x = random.randint(6, ((self.w - BLOCK_SIZE) // BLOCK_SIZE) - (2)) * BLOCK_SIZE
            y = random.randint(2, ((self.h - BLOCK_SIZE) // BLOCK_SIZE) - (6 + 2)) * BLOCK_SIZE
            
            # x = 160
            # y = 60

            # Define the first and second sides of the window (vertical)
            self.first_side = Point(x, y)
            self.second_side = Point(x, y + (6 * BLOCK_SIZE))

        # Check if the Quad (player) overlaps with the window
        # If so, place the window again (recursion)
        if self.Quad == self.first_side or self.Quad == self.second_side:
            self.place_window()

    def is_collision_wall(self):
        """
        Check if the Quad (player) has collided with the window or the boundaries of the game screen.
        Returns True if a collision occurs, otherwise False.
        """
        # Check for collision with the window or boundaries
        if (
            self.Quad.x > self.w - BLOCK_SIZE  # Check if Quad hits the right boundary
            or self.Quad.x < 0  # Check if Quad hits the left boundary
            or self.Quad.y > self.h - BLOCK_SIZE  # Check if Quad hits the bottom boundary
            or self.Quad.y < 0  # Check if Quad hits the top boundary
        ):
            return True  # Collision detected
        return False  # No collision detected

    def is_collision_Window(self):
        """
        Check if the Quad (player) has collided with the window.
        Returns True if a collision occurs, otherwise False.
        """
        # Check for collision with the window
        if self.Quad == self.first_side or self.Quad == self.second_side:
            return True  # Collision detected
        return False  # No collision detected

    def _update_ui(self):
        if not self.render:
            return  # Skip the rendering if it's disabled

        self.display.fill(BLACK)  # Clear the screen

        # Draw the Quad (player) in the center
        pygame.draw.rect(
            self.display,
            BLUE1,
            pygame.Rect(self.Quad.x, self.Quad.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display, BLUE2, pygame.Rect(self.Quad.x + 4, self.Quad.y + 4, 12, 12)
        )

        # Draw the 4 diagonal spokes (rotated 45 degrees)
        spoke_length = 15  # Length of each spoke
        spoke_width = 2    # Width of the spokes

        # Center of the Quad (its current position)
        center_x = self.Quad.x + BLOCK_SIZE // 2
        center_y = self.Quad.y + BLOCK_SIZE // 2

        pygame.draw.line(self.display, WHITE, (center_x, center_y), 
                        (center_x - spoke_length, center_y - spoke_length), spoke_width)
        pygame.draw.line(self.display, WHITE, (center_x, center_y), 
                        (center_x + spoke_length, center_y - spoke_length), spoke_width)
        pygame.draw.line(self.display, WHITE, (center_x, center_y), 
                        (center_x - spoke_length, center_y + spoke_length), spoke_width)
        pygame.draw.line(self.display, WHITE, (center_x, center_y), 
                        (center_x + spoke_length, center_y + spoke_length), spoke_width)

        # Draw the window
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.first_side.x, self.first_side.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.second_side.x, self.second_side.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        # Update the display
        pygame.display.flip()


    def _move(self, action):
        """
        Update the direction of the Quad's movement based on the action taken.
        Parameters:
            action (list): A one-hot encoded list indicating the direction of movement:
                           [straight, right, left]
        """
        # List of directions in clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        # Get the index of the current direction
        idx = clock_wise.index(self.direction)

        # Determine the new direction based on the action
        if action == self.STRAIGHT:
            # Action [1, 0, 0] means moving straight, so the direction doesn't change
            new_dir = clock_wise[idx]  # No change in direction
        elif action == self.RIGHT:
            # Action [0, 1, 0] means turning right
            next_idx = (
                idx + 1
            ) % 4  # Get the index for the next direction in the clockwise list
            new_dir = clock_wise[next_idx]  # Update direction to the right
        elif action == self.LEFT:
            # Action [0, 0, 1] means turning left
            next_idx = (
                idx - 1
            ) % 4  # Get the index for the next direction in the counter-clockwise list
            new_dir = clock_wise[next_idx]  # Update direction to the left

        # Update the Quad's direction
        self.direction = new_dir
        # Update the position of the Quad based on the new direction
        x = self.Quad.x  # Current x-coordinate of the Quad
        y = self.Quad.y  # Current y-coordinate of the Quad

        # Move the Quad based on the current direction
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE  # Move the Quad to the right
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE  # Move the Quad to the left
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE  # Move the Quad down
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE  # Move the Quad up

        # Set the new Quad position
        self.Quad = Point(x, y)

    #! Observation
    def state(self):
        """
        Get the current state of the game.

        Returns:
        numpy array: Current state including direction and position of the Quad.
        """
        self.middle_of_the_window = Point(0, 0)
        if self.option == "0" :
            self.middle_of_the_window = Point(self.first_side.x + BLOCK_SIZE * 3, self.first_side.y )        
        elif self.option == "1" :
            self.middle_of_the_window = Point(self.first_side.x, self.first_side.y + BLOCK_SIZE * 3)

        previous_state = self.memory.states[-1]  # Get the last state from memory
        previous_quad_x = previous_state[0]  # Get the previous x-coordinate of the Quad
        previous_quad_y = previous_state[1]
        # Map the direction to a numerical value
        direction_map = {
            Direction.RIGHT: 0,  # Assign 0 for RIGHT direction
            Direction.DOWN: 1,   # Assign 1 for DOWN direction
            Direction.LEFT: 2,   # Assign 2 for LEFT direction
            Direction.UP: 3      # Assign 3 for UP direction
        }

        # Convert the current direction to its corresponding numeric value
        direction = direction_map[self.direction]

        # Create a numpy array to hold the state including direction and position (x, y)
        state = np.array(
            [float(self.Quad.x), float(self.Quad.y), (direction),
            float(previous_quad_x), float(previous_quad_y),
            self.middle_of_the_window.x > self.Quad.x,
            self.middle_of_the_window.x < self.Quad.x,
            self.middle_of_the_window.y > self.Quad.y,
            self.middle_of_the_window.y < self.Quad.y
            ], dtype=np.float32
        )

        # Return the state as a numpy array
        return state


    def actions(self):
        """
        Get the number of possible actions.

        Returns:
            list: A list of possible actions (STRAIGHT, LEFT, RIGHT).
        """
        self.STRAIGHT = 0  # Action to continue moving straight
        self.RIGHT = 1     # Action to turn right
        self.LEFT = 2      # Action to turn left
        actions = [self.STRAIGHT, self.RIGHT, self.LEFT]  # List of possible actions
        return actions  # Return the possible actions list

    #!Reward function
    def step(self, action):
        """
        Perform one step in the environment based on the selected action.

        Parameters:
            action (list): A one-hot encoded list indicating the direction of movement.

        Returns:
            next_state (numpy array): The state after performing the action.
            reward (int): The reward for the action taken.
            done (bool): Whether the episode has ended.
        """

        # Process user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()  # Exit the program if the close button is clicked

        self.frame_iteration += 1  # Increment the frame iteration count
        self.memory.add_state(self.state())  # Add the current state to the memory

        self._move(action)  # Move the Quad based on the selected action
        self.reward = 0  # Initialize the reward
        self.done = False  # Initialize the done status

        # Previous state information
        previous_state = self.memory.states[-1]  # Get the last state from memory
        previous_quad_x = previous_state[0]  # Get the previous x-coordinate of the Quad

        #!# Positive Reward: Check if the Quad moves straight through a specific window range

        if (
            (self.first_side.y < self.Quad.y < self.second_side.y) and  # Check if Quad is between window sides
            (previous_quad_x < self.Quad.x) and  # Ensure the Quad is moving forward
            (action == self.STRAIGHT)  # Action is to move straight
        ):
            self.reward += 1  # Basic reward for moving straight within the window range
            # Additional rewards for passing through specific zones in the window
            if (
                (self.first_side.y + (2 * BLOCK_SIZE) <= self.Quad.y <= self.first_side.y + (3 * BLOCK_SIZE)) or
                (self.second_side.y - (2 * BLOCK_SIZE) <= self.Quad.y < self.second_side.y - (1 * BLOCK_SIZE))
            ):
                self.reward += 2  # Reward for moving through these special zones
                if (
                    self.second_side.y - (3 * BLOCK_SIZE)
                    <= self.Quad.y
                    < self.second_side.y - (2 * BLOCK_SIZE)
                ):
                    self.reward += 3  # Maximum reward for moving through the center zone
        # Reward for reaching a specific point in the window
        if (
            self.first_side.y < self.Quad.y < self.second_side.y and  # Quad is within window bounds
            (action == self.STRAIGHT) and  # Action is to move straight
            (self.first_side.x -BLOCK_SIZE < self.Quad.x <= self.first_side.x) and  # Quad is aligned with the first window side
            (self.first_side.x - BLOCK_SIZE == previous_quad_x) and  # Previous position was just outside the window
            (previous_quad_x < self.Quad.x)  # Ensure the Quad is moving forward

        ):
            self.reward += 5  # High reward for correctly passing through
            self.done = False  # Keep the episode running
        # Reward for moving straight through the window's x-coordinate
        if (
            (self.first_side.y < self.Quad.y < self.second_side.y) and  # Quad is within window bounds
            (action == self.STRAIGHT) and  # Action is to move straight
            (self.first_side.x < self.Quad.x <= self.first_side.x + BLOCK_SIZE) and
            (self.first_side.x == previous_quad_x) and
             (previous_quad_x < self.Quad.x)  # Ensure the Quad is moving forward

        ):
            self.reward += 15  # High reward for correct movement through the window
            if (
                (self.first_side.y + (2 * BLOCK_SIZE) <= self.Quad.y <= self.first_side.y + (3 * BLOCK_SIZE)) or
                (self.second_side.y - (2 * BLOCK_SIZE) <= self.Quad.y < self.second_side.y - (1 * BLOCK_SIZE))
            ):
                self.reward += 20  # Extra reward for passing through specific zones in the window
                if (
                    self.second_side.y - (3 * BLOCK_SIZE)
                    <= self.Quad.y
                    < self.second_side.y - (2 * BLOCK_SIZE)
                ):
                    self.reward += 30  # High reward for passing exactly through the center
            self.done = True  # End the episode

        #!# Negative Reward: Penalize for collisions
        elif self.is_collision_wall():
            self.reward += -15  # Penalize for hitting the wall
            self.done = True  # End the episode

        elif self.is_collision_Window():
            self.reward += -5  # Penalize for hitting the window
            self.done = True  # End the episode

        self._update_ui()  # Update the game display
        self.clock.tick(SPEED)  # Control the game speed

        next_state = self.state()  # Get the next state after the action
        return next_state, self.reward, self.done  # Return the next state, reward, and done status