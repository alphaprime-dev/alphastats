#!/bin/bash

set -ex

# prek hook install
uv run prek install --overwrite --hook-type pre-commit --hook-type commit-msg
