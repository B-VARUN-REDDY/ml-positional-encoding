# Contributing to My Project

Thanks for checking out my positional encoding implementation! I built this as a personal project/technical assessment, but I'm open to suggestions or improvements.

## Development Setup

If you want to run this locally, here's how I set up my environment:

1. **Clone the repo**
   ```bash
   git clone https://github.com/B-VARUN-REDDY/ml-positional-encoding.git
   ```

2. **Create virtual env**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8 black  # dev tools
   ```

## Testing

I've included a comprehensive test suite. Before submitting any changes, please ensure all tests pass:

```bash
python tests/test_positional_encoding.py
```

## detailed Style Guide

I try to adhere to standard Python patterns:
- Type hints for all function arguments/returns
- Docstrings for classes and functions
- 4-space indentation
- `black` for formatting

## Project Structure

- `src/`: Core implementation code
- `tests/`: Unit and integration tests
- `scripts/`: Helper scripts for training and visualization
- `experiments/`: Where training artifacts are saved

---

**Varun Reddy**
