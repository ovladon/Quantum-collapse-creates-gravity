#!/usr/bin/env python3
"""
Module: utils.py
Provides common utility functions for the emergent gravity simulation project.
"""

import logging

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

