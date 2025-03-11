# LLM-Powered Billiards Game

A simplified pool game built with Python and Pygame, where an LLM (Llama-3.2-3B-Instruct or Qwen2.5-7B-Instruct) is used to determine the shot angle and power. The AI plays by analyzing ball positions and choosing the best possible shot based on pre-defined rules.

## About the Experiment

This project explores how a language model (LLM) can interact with a numerical and spatial environment by predicting the best shot in a billiards game. It uses:

- **Physics Simulation**: Ball movement, friction, collisions, and pocketing.
- **Prompt Engineering**: Spoon-feeding the LLM with precise details about the game state.
- **Local API Calls**: The LLM is hosted locally, responding with JSON containing angle and power values.

📖 **Read the full blog post here:** [How I Tricked an LLM into Playing a Pool Game](https://medium.com/@palashm0002/how-i-tricked-an-llm-into-playing-a-very-simplified-pool-game-7c44d858ae61)

## Features

- ✅ **Simple 2D Pool Game** using Python & Pygame  
- ✅ **LLM-Assisted Gameplay** - AI chooses shot angles & power  
- ✅ **Pocket Detection & Foul Rules** - If the cue ball enters a pocket, it's reset  
- ✅ **Spoon-fed AI Inputs** - Distance & angles of all balls & pockets  
- ✅ **Fallback Logic** - If LLM gives invalid output, use closest-ball logic  
- ✅ **Max 10 Shots Per Game** - If AI fails to pocket the black ball, it loses  

## How It Works

1. The cue ball is placed on the table along with other colored balls and a black ball.
2. The game simulates **ball movement, friction, and collisions**.
3. Once the balls stop moving, the **LLM is queried** for the best shot.
4. The LLM receives:
   - Positions & angles of all balls
   - Distances & angles from pockets
   - The closest ball to the cue ball (fallback shot)
5. The LLM returns **JSON with an angle and power for the shot**.
6. The cue ball moves based on AI input, and the cycle repeats **until the black ball is potted or the 10-shot limit is reached**.

## Setup & Installation

### Prerequisites

- Python **3.8+**
- **Pygame**
- A locally hosted **LLM (Llama-3.2 or Qwen2.5)**

### Install Dependencies

```bash
pip install pygame requests
