# Contributing to CVAE Plant Stress Detector

Thank you for your interest in contributing to the **CVAE Plant Stress Detector** project! We welcome contributions from machine learning researchers, computer vision engineers, and open-source practitioners.

---

## Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to [ravvanagarjun@gmail.com](mailto:ravvanagarjun@gmail.com).

---

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ravva-78/cvae-plant-stress.git
cd cvae-plant-stress
```

### 2. Set Up a Virtual Environment
```bash
# Using venv (Python 3.10+)
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black
```

### 4. Verify Local Environment
Run the test suite and fast zero-setup demo:
```bash
pytest -v
python demo.py --test-run
```

---

## Branching & Commit Conventions

### Branch Naming Policy
- `feature/<short-description>`: New architecture features, layers, or loss functions.
- `fix/<short-description>`: Bug fixes or stability patches.
- `perf/<short-description>`: Inference acceleration, quantization, or memory optimization.
- `docs/<short-description>`: Documentation, diagram updates, or README enhancements.
- `refactor/<short-description>`: Code restructuring without functional alterations.

### Conventional Commits
All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
```
<type>(<scope>): <concise imperative summary>

[optional body explaining architectural rationale]

[optional footer(s)]
```

**Allowed Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.

**Examples:**
- `feat(model): add beta-VAE KL scaling parameter to cvae_loss`
- `fix(webapp): resolve base64 serialization for single-channel heatmaps`
- `test(shapes): add contract tests for latent dimension bottleneck`

---

## Developer Workflows (Makefile)

If `make` is installed on your system:
```bash
make test       # Run pytest test suite with coverage
make lint       # Run flake8 style validation
make demo       # Execute zero-setup CLI demonstration
make webapp     # Launch Flask interactive demo on port 5000
make clean      # Clean Python byte-caches and build artifacts
```

---

## Pull Request Guidelines

1. **Keep PRs Focused:** Limit changes to a single logical feature, bug fix, or refactor.
2. **Add Unit Tests:** Every functional change to `utils/model.py` or `webapp/app.py` must include corresponding tests under `tests/`.
3. **Verify Green Tests:** Ensure `pytest -v` passes locally and in the GitHub Actions CI matrix across Python 3.10, 3.11, and 3.12.
4. **Document Changes:** Update docstrings, type annotations, and README if CLI flags or API schemas change.

Thank you for helping push forward agricultural deep generative vision!
