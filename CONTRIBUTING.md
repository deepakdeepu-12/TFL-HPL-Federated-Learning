# Contributing to TFL-HPL

Thank you for your interest in contributing to the TFL-HPL project!

## How to Contribute

### 1. Fork the Repository

```bash
git clone https://github.com/deepakdeepu-12/TFL-HPL-Federated-Learning.git
cd TFL-HPL-Federated-Learning
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

Ensure your code:
- Follows PEP 8 style guide
- Includes type hints
- Has comprehensive docstrings
- Passes all existing tests
- Includes new tests for new functionality

### 4. Test Locally

```bash
# Run tests
pytest tests/ -v

# Check code style
flake8 tfl_hpl/
black tfl_hpl/

# Type checking
mypy tfl_hpl/
```

### 5. Submit Pull Request

- Include descriptive PR title and description
- Reference related issues
- Provide before/after comparisons if applicable
- Add yourself to contributors list

## Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install package in development mode
pip install -e .
```

## Code Style

- **Formatting**: Use `black tfl_hpl/`
- **Linting**: Use `flake8 tfl_hpl/`
- **Type Hints**: All functions must have type hints
- **Docstrings**: Use Google-style docstrings

## Commit Message Guidelines

```
[type] Brief description (max 50 chars)

Detailed explanation (max 72 chars per line)

- Bullet point 1
- Bullet point 2

Issues: #123, #456
```

**Types**: feat, fix, docs, test, refactor, perf, style

## Testing

- Add tests for all new features
- Maintain >80% code coverage
- Test edge cases and error conditions
- Include integration tests where applicable

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update CHANGELOG.md

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

---

Thank you for contributing!
