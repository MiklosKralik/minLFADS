# minLFADS
minLFADS is a minimal implementation of LFADS (Latent Factor Analysis via Dynamical Systems) in PyTorch for fun.
It also uses leaky RNNs instead of GRUs because why not.

## Installation
```bash
git clone https://github.com/MiklosKralik/minLFADS.git
pip install -e .
```

## Train LFADS model on macaque reaching data
```python
python projects/reaching/reaching.py
```
