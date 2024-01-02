# Contributing to `MidiTok`

- Reporting a bug.
- Discussing the current state of the code.
- Submitting a fix.
- Proposing new features.
- Becoming a maintainer.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests

Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). We actively welcome your pull requests:

1. Fork the repo and create your branch from the `main`.
2. If you've added code that should be tested, add [tests](tests).
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Report bugs using Github's [issues](https://github.com/Natooz/MidiTok/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/Natooz/MidiTok/issues/new).

## Write bug Reports with Detail, Background, and Sample Code

**Great Bug Reports** tend to have:

- A quick summary and/or background.
- Steps to reproduce.
  - Be specific!
  - Give a sample code if you can, for example,
- What you expected would happen.
- What actually happens.
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work).

## Development

### Tests

We use `pytest`/`pytest-xdist` for testing and `pytest-cov` for measuring coverage. Running all the tests can take between 10 to 30min depending on your hardware. You don't need to run all of them, but try to run those affected by your changes.

```bash
pip install pytest-cov "pytest-xdist[psutil]"
pytest --cov=./ --cov-report=xml -n auto --durations=0 -v tests/
```

### Use a Consistent Coding Style

We use the [ruff](https://github.com/astral-sh/ruff) formatter for Python in this project. Ruff allows to automatically analyze the code and format it according to rules if needed. This is handled by using pre-commit (following section).

### Pre-commit Lints

Linting is configured via [pre-commit](https://www.pre-commit.com/). You can set up pre-commit by running:

```bash
pip install pre-commit
pre-commit install  # installs pre-commit Git hook in the repository
```

When your changes are finished and the tests are passing, you can run `pre-commit run` to check if your code lints according to our ruff rules.
If errors are found, we encourage you to fix them to follow the best code practices. If you struggle with this step, don't hesitate to ask for help, and to even commit and push anyway. Contributors will be able to help you.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
