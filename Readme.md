## Wafer fault detection

This repository contains processing pipelines for the automated detection of faults such as delaminations, discontinuities, dusts and short circuits in wafer images.

## Setup

### Step 1: clone the repository
```
git clone git@github.com:fkroeber/wafer_imaging.git
cd wafer_imaging
```

### Step 2: create virtual environment

### Step 2a: ...via venv

For Windows
```
python -m venv wafer
wafer\Scripts\activate
pip install --upgrade pip
pip install -r reqs_windows.txt
```

For Linux/macOS
```
python -m venv wafer
source wafer/bin/activate
pip install --upgrade pip
pip install -r reqs_unix.txt
```

### Step 2b: ...via conda
```
conda create -n wafer python=3.10
conda activate wafer
pip install -r reqs_unix.txt
```