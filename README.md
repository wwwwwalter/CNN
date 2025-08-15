# MNIST CNN (PyTorch)

A minimal, runnable example of training a CNN on MNIST using PyTorch.

## Install dependencies (CPU)

```bash
python3 -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Train

```bash
python3 /workspace/mnist_cnn.py --epochs 3 --batch-size 128 --lr 1e-3 --data-dir /workspace/data
```

- Use `--device cuda` to force GPU if available, or leave default `auto`.
- To avoid downloading (offline), place MNIST under `/workspace/data` and add `--no-download`.

The best model checkpoint will be saved to `/workspace/artifacts/mnist_cnn_best.pt`.