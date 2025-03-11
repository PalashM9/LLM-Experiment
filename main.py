import pygame
import math
import sys
import random
import requests
import json
import re

# Window dimensions
WIDTH, HEIGHT = 800, 400

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
YELLOW = (255, 215, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)

BALL_RADIUS = 12
POCKET_RADIUS = 25
FRICTION = 0.98
MAX_SHOTS = 10

################################################################################
# BALL CLASS
################################################################################
class Ball:
    def __init__(self, x, y, color, is_cue=False, is_black=False):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.color = color
        self.is_cue = is_cue
        self.is_black = is_black

    def update(self):
        """Updates the ball's position based on velocity and apply friction."""
        self.x += self.vx
        self.y += self.vy

        # Apply friction
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Collides with table edges
        if self.x - BALL_RADIUS < 0:
            self.x = BALL_RADIUS
            self.vx = -self.vx
        elif self.x + BALL_RADIUS > WIDTH:
            self.x = WIDTH - BALL_RADIUS
            self.vx = -self.vx

        if self.y - BALL_RADIUS < 0:
            self.y = BALL_RADIUS
            self.vy = -self.vy
        elif self.y + BALL_RADIUS > HEIGHT:
            self.y = HEIGHT - BALL_RADIUS
            self.vy = -self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), BALL_RADIUS)

    def is_moving(self):
        """Return True if velocity is above a small threshold."""
        return abs(self.vx) > 0.05 or abs(self.vy) > 0.05

################################################################################
# COLLISION HANDLING
################################################################################
def handle_collisions(balls):
    """
    Check collisions pairwise among all balls in the list.
    """
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            b1 = balls[i]
            b2 = balls[j]
            dx = b2.x - b1.x
            dy = b2.y - b1.y
            dist = math.hypot(dx, dy)

            if dist == 0:
                dist = 0.001

            if dist < 2 * BALL_RADIUS:
                # Overlap
                overlap = 2 * BALL_RADIUS - dist
                push_x = (dx / dist) * (overlap / 2)
                push_y = (dy / dist) * (overlap / 2)

                b1.x -= push_x
                b1.y -= push_y
                b2.x += push_x
                b2.y += push_y

                # Simple elastic collision
                nx = dx / dist
                ny = dy / dist
                vn1 = b1.vx * nx + b1.vy * ny
                vn2 = b2.vx * nx + b2.vy * ny

                b1.vx += (vn2 - vn1) * nx
                b1.vy += (vn2 - vn1) * ny
                b2.vx += (vn1 - vn2) * nx
                b2.vy += (vn1 - vn2) * ny

################################################################################
# HELPER: ANGLE / DISTANCE
################################################################################
def angle_degrees(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return math.degrees(math.atan2(dy, dx))

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

################################################################################
# LLM CALL
################################################################################
def call_llm(prompt_text, fallback_angle):
    """
    Calls local LLM with temperature=0, looks for JSON {"angle_degrees": X, "power": Y}.
    If invalid or out-of-range, fallback to (fallback_angle, 8.0).
    """
    data = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.0,   # minimize randomness
        "max_tokens": -1,
        "stream": False
    }

    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        result = response.json()

        print("---------- LLM RAW RESPONSE ----------")
        print(json.dumps(result, indent=2))
        print("--------------------------------------")

        content = result["choices"][0]["message"]["content"]
        print("\nLLM responded with text:\n", content)

        # Strict regex parse for JSON
        match = re.search(
            r'\{\s*"angle_degrees"\s*:\s*([\-\d\.]+)\s*,\s*"power"\s*:\s*([\-\d\.]+)\s*\}',
            content
        )
        if not match:
            print("No valid JSON found in LLM output. Falling back.")
            return fallback_angle, 8.0

        angle_str, power_str = match.groups()
        angle_deg = float(angle_str)
        power_val = float(power_str)

        # Normalize angle into [0, 360)
        angle_deg %= 360

        # If after normalization, angle still not in [0,360) or power out of [0..15], fallback
        if not (0 <= angle_deg < 360) or not (0 <= power_val <= 15):
            print("LLM angle/power out of range. Falling back to closest-ball angle.")
            return fallback_angle, 8.0

        return angle_deg, power_val

    except Exception as e:
        print("Error calling/parsing LLM. Fallback to closest-ball angle.\n", e)
        return fallback_angle, 8.0

################################################################################
# PROMPT BUILDER
################################################################################
def build_prompt_for_llm(
    cue_ball, all_balls, pockets, shots_taken, fallback_angle, fallback_target_color
):
    """
    Spoon-fed prompt: Includes each ball's angle/distance to each pocket, so the LLM
    can consider potential direct pocket shots. We highlight the closest ball, but also
    show pocket info for all balls.

    We strongly emphasize that angle_degrees must be in [0..359] and power in [0..15].
    """
    lines = []
    lines.append("You are a billiards AI in a single-player game with multiple balls.")
    lines.append("The black ball is the final target, but there are other object balls too.")
    lines.append(f"Shots taken: {shots_taken}/{MAX_SHOTS}. Table size: width={WIDTH}, height={HEIGHT}.")
    lines.append("")
    lines.append("Pockets (corners):")
    for i, (px, py) in enumerate(pockets, start=1):
        lines.append(f"  Pocket {i}: (x={px}, y={py})")
    lines.append("")

    lines.append(f"Cue Ball at (x={cue_ball.x:.2f}, y={cue_ball.y:.2f}).")
    lines.append(f"Velocity: (vx={cue_ball.vx:.2f}, vy={cue_ball.vy:.2f}).")
    lines.append("")

    # Identify the closest ball for fallback
    dist_to_closest = float('inf')
    closest_ball_color = None
    for b in all_balls:
        if b.is_cue:
            continue
        dist_b = distance(cue_ball.x, cue_ball.y, b.x, b.y)
        if dist_b < dist_to_closest:
            dist_to_closest = dist_b
            closest_ball_color = "black" if b.is_black else str(b.color)

    # Summarize ball info, including distance/angle to each pocket
    lines.append("BALLS INFO (for each ball, also show distance/angle to pockets):")
    idx = 1
    for b in all_balls:
        if b.is_cue:
            continue
        dist_b = distance(cue_ball.x, cue_ball.y, b.x, b.y)
        ang_b = angle_degrees(cue_ball.x, cue_ball.y, b.x, b.y)
        c_str = "black" if b.is_black else str(b.color)

        lines.append(f"Ball {idx}: color={c_str}, (x={b.x:.2f}, y={b.y:.2f})")
        lines.append(f"  Distance from cue={dist_b:.2f}, angle from cue={ang_b:.2f}")

        # For each pocket, also compute distance from this ball to the pocket
        # and angle from the ball to the pocket:
        lines.append("  Pocket distances/angles:")
        for i, (px, py) in enumerate(pockets, start=1):
            dist_pocket = distance(b.x, b.y, px, py)
            ang_pocket = angle_degrees(b.x, b.y, px, py)
            lines.append(f"    Pocket {i}: dist={dist_pocket:.2f}, angle={ang_pocket:.2f}")

        lines.append("")  # blank line after each ball
        idx += 1

    lines.append(f"The fallback angle is {fallback_angle:.2f} degrees toward the closest ball ({closest_ball_color}).")
    lines.append("")
    lines.append("IMPORTANT: angle_degrees must be in [0..359], power must be in [0..15].")
    lines.append('Please respond with STRICT JSON: {"angle_degrees": <float>, "power": <float>}')
    lines.append("No extra text, just valid JSON. Good luck!")

    return "\n".join(lines)

################################################################################
# MAIN
################################################################################
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LLM Billiards (Pocket Info Included)")

    clock = pygame.time.Clock()

    # Create the cue ball
    cue_ball = Ball(WIDTH // 4, HEIGHT // 2, WHITE, is_cue=True)
    # Create the black ball
    black_ball = Ball(3 * WIDTH // 4, HEIGHT // 2, BLACK, is_black=True)

    # Extra colored balls
    colors = [RED, BLUE, YELLOW, MAGENTA, CYAN]
    object_balls = []
    for color in colors:
        x_offset = (3 * WIDTH // 4) + random.randint(-40, 40)
        y_offset = (HEIGHT // 2) + random.randint(-40, 40)
        object_balls.append(Ball(x_offset, y_offset, color))

    # Combine
    balls = [cue_ball, black_ball] + object_balls

    pockets = [
        (0, 0),
        (WIDTH, 0),
        (0, HEIGHT),
        (WIDTH, HEIGHT)
    ]

    shots_taken = 0
    game_over = False
    win_message = None

    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not game_over:
            # 1) Update positions
            for b in balls:
                b.update()

            # 2) Collisions
            handle_collisions(balls)

            # 3) Check pockets
            removed_balls = []
            for b in balls:
                for (px, py) in pockets:
                    distp = distance(b.x, b.y, px, py)
                    if distp < POCKET_RADIUS:
                        # Black ball potted => immediate win
                        if b.is_black:
                            game_over = True
                            win_message = "Black ball potted! AI wins!"
                        # Cue ball potted => foul: reposition
                        elif b.is_cue:
                            print("Foul: Cue ball pocketed. Resetting to center.")
                            b.x = WIDTH // 4
                            b.y = HEIGHT // 2
                            b.vx = 0
                            b.vy = 0
                        else:
                            # Remove other ball
                            removed_balls.append(b)

            for rb in removed_balls:
                if rb in balls:
                    balls.remove(rb)

            # 4) If all balls are at rest, get next shot from LLM
            if not any(b.is_moving() for b in balls):
                if shots_taken < MAX_SHOTS and not game_over:
                    shots_taken += 1
                    print(f"\n----- Shot {shots_taken} of {MAX_SHOTS} -----")

                    # Find the closest ball to the cue ball for fallback
                    dist_closest = float('inf')
                    closest_ball = None
                    for b in balls:
                        if not b.is_cue:
                            d = distance(cue_ball.x, cue_ball.y, b.x, b.y)
                            if d < dist_closest:
                                dist_closest = d
                                closest_ball = b

                    if closest_ball:
                        fallback_angle = angle_degrees(cue_ball.x, cue_ball.y, closest_ball.x, closest_ball.y)
                        fallback_target_color = "black" if closest_ball.is_black else str(closest_ball.color)
                    else:
                        # If there's literally no other ball, fallback to 0
                        fallback_angle = 0.0
                        fallback_target_color = "???"

                    # Build prompt
                    prompt_text = build_prompt_for_llm(
                        cue_ball, balls, pockets, shots_taken,
                        fallback_angle, fallback_target_color
                    )

                    # Call LLM
                    angle_deg, power_val = call_llm(prompt_text, fallback_angle)

                    # Apply shot
                    rad = math.radians(angle_deg)
                    cue_ball.vx = math.cos(rad) * power_val
                    cue_ball.vy = math.sin(rad) * power_val
                else:
                    # No more shots or game ended
                    if not game_over:
                        game_over = True
                        if not win_message:
                            # If black ball is still around => lose
                            if any(b.is_black for b in balls):
                                win_message = "No more shots. Black ball remains. AI loses!"
                            else:
                                win_message = "All balls cleared or black ball potted earlier."

        # Draw
        screen.fill(GREEN)
        for (px, py) in pockets:
            pygame.draw.circle(screen, GRAY, (px, py), POCKET_RADIUS)

        for b in balls:
            b.draw(screen)

        font = pygame.font.SysFont(None, 24)
        text_shots = font.render(f"Shots: {shots_taken}/{MAX_SHOTS}", True, WHITE)
        screen.blit(text_shots, (10, 10))

        if game_over and win_message:
            text = font.render(win_message, True, WHITE)
            screen.blit(text, (WIDTH // 2 - 150, HEIGHT // 2))

        pygame.display.flip()

if __name__ == "__main__":
    main()
