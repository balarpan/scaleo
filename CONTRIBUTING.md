# Contributing to project

Thank you for your interest in contributing! We welcome all contributions, including bug fixes, documentation updates, and feature requests. 

## Table of Contents
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Suggest a Feature](#how-to-suggest-a-feature)
- [Pull Request Process](#pull-request-process)
- [Coding Style Guidelines](#coding-style-guidelines)

---

## How to Report a Bug
Before submitting a bug report, please check our existing issues to see if it has already been reported. If not, open a new issue and include:
* A clear, descriptive title.
* Expected vs. actual behavior.
* Detailed steps to reproduce the issue.
* Code snippets, log outputs, or screenshots if applicable.
* Environment details (OS, language version, package version).

## How to Suggest a Feature
We love new ideas! To request a feature, open a new issue and provide:
* A summary of the feature you want.
* An explanation of why this feature is useful or necessary.
* Examples of how the feature would work or look.

### Local Setup
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/balarpan/scaleo.git
   ```
3. Create a descriptive branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dependencies:
   ```bash
   uv sync
   ```

## Pull Request Process
1. Ensure any installation or build steps are complete.
2. Update the `README.md` or relevant documentation if your change introduces new functionality.
3. Write clean code and include tests if applicable.
4. Push your branch to GitHub and open a Pull Request targeting the `dev` branch.
5. Provide a clear explanation of your changes in the PR description.
6. A maintainer will review your code. Address any feedback as requested.

## Coding Style Guidelines
* **Code Style**: PEP 8 with [Google style docstring](https://google.github.io/styleguide/pyguide.html).
* **Commit Messages**: Use clear, concise commit messages in the imperative mood (e.g., `Fix image size bug`).

