---
name: git
description: Git operations: branching, commit message format, pre-commit hook handling, and push policy
---

# Git Instructions

## Core Rule

**Never commit or push directly to `main`.** All changes must go through a Pull Request on a feature branch.

## Branching

When creating a new branch off `main`, always sync local `main` with the remote first:

```sh
git checkout main && git pull upstream main
git checkout -b your-new-branch
```

## Committing

**Only commit when explicitly requested by the user.**

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit. Claude should handle failures without asking the user:

1. Attempt `git commit` — hooks run automatically.
2. If hooks fail, they will try to auto-fix (formatting, trailing whitespace, etc.).
3. If auto-fixed: `git add <modified-files>` then retry the same commit message.
4. If not auto-fixed: manually adjust per the error message, then retry.
5. Only notify the user if a manual code fix is required that cannot be automated.

## Pushing

**Never auto-push.** Only run `git push` when explicitly asked by the user.
