# Contributing to Skin Lesion Classification

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/skin-lesion-classification.git
   cd skin-lesion-classification
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/Yasir-Alazmi/skin-lesion-classification.git
   ```

---

## Development Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

---

## Making Changes

1. Create a **feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, following the [Code Style](#code-style) guidelines.
3. Write or update tests where applicable.
4. Commit with a meaningful message using [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(model): add DenseNet-121 backbone
   fix(dataset): handle missing image files gracefully
   docs(readme): add inference example
   ```
5. Push and open a Pull Request against `main`.

---

## Pull Request Process

- Ensure your branch is up to date with `main` before opening a PR.
- Fill in the PR template completely.
- PRs require at least one review before merging.
- All CI checks (linting, tests) must pass.

---

## Code Style

This project follows **PEP 8** with the following conventions:

| Tool | Purpose | Config |
|------|---------|--------|
| `black` | Auto-formatter | `pyproject.toml` |
| `isort` | Import sorting | `pyproject.toml` |
| `flake8` | Linting | `pyproject.toml` |

Run formatting before committing:
```bash
black src/
isort src/
flake8 src/
```

---

## Reporting Issues

- Use the [GitHub Issues](https://github.com/Yasir-Alazmi/skin-lesion-classification/issues) page.
- Include: Python version, OS, GPU/CUDA info, and a minimal reproducible example.
- Label your issue appropriately (`bug`, `enhancement`, `question`).

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
