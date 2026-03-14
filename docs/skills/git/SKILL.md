---
name: git
description: Git operations: branching, commit message format, pre-commit hook handling, and push policy
---

# Git Instructions

## Branching

When creating a new branch off `main`, always sync local `main` with the remote first:

```sh
git checkout main && git pull upstream main
git checkout -b your-new-branch
```

## Committing

**Only commit when explicitly requested by the user.**

### Commit Message Format

Always include a detailed summary of the conversation:

```
<brief description of changes>

Changes made:
- (AI): <concise summary of what Claude did>
- (User): <manual edits or user-authored code, if any>
- (AI): <additional Claude action if substantially different>

User raw prompts:
- <exact user prompt 1>
- <exact user prompt 2>

Co-Authored-By: Claude <noreply@anthropic.com>
```

Guidelines:
- **Changes made**: Keep it concise, reduce to the most important bullet points.
- **User raw prompts**: Use the exact prompt. For long ones (error messages, code blocks), replace with descriptive placeholders like `<ImportError: ...>` or `<code block in file.py>`.
- Do not add "Generated with Claude Code" — Co-Authored-By is enough.
- To detect user vs Claude changes: compare `git diff` against Claude's tool actions. Anything not from Claude's tool usage is a user change. Use `<unknown>` if unsure.

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit. Claude should handle failures without asking the user:

1. Attempt `git commit` — hooks run automatically.
2. If hooks fail, they will try to auto-fix (formatting, trailing whitespace, etc.).
3. If auto-fixed: `git add <modified-files>` then retry the same commit message.
4. If not auto-fixed: manually adjust per the error message, then retry.
5. Only notify the user if a manual code fix is required that cannot be automated.

## Pushing

**Never auto-push.** Only run `git push` when explicitly asked by the user.
