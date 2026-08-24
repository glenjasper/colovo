#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def display_path(path, folder):
    parts = path.split(os.sep)

    try:
        index = parts.index(folder)
        p = os.sep.join(parts[index:])
    except:
        p = path

    return p
