#!/bin/bash

docker builder prune && docker system prune -a && docker buildx prune --all
