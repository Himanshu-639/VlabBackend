#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Download necessary NLTK resources
python nltk_downloader.py
