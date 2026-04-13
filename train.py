#!/usr/bin/env python3
"""
Train a SpaceAce agent.

Usage:
    python train.py --level 0 --timesteps 500000
    python train.py --level 1 --timesteps 1000000 --resume models/0/best_model
"""

from spaceace.agents.ppo.train import main

if __name__ == "__main__":
    main()
