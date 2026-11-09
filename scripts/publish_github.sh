#!/usr/bin/env bash
# Publie le dépôt sur GitHub si la CLI `gh` est authentifiée.
# Usage : bash scripts/publish_github.sh [nom-du-repo]
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_NAME="${1:-deploiement-ia-tp}"

if ! command -v gh >/dev/null 2>&1; then
  echo "La CLI GitHub 'gh' n'est pas installée."
  echo "Installez-la (https://cli.github.com/) puis : gh auth login"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Vous n'êtes pas authentifié. Lancez : gh auth login"
  exit 1
fi

git add -A
if ! git diff --cached --quiet; then
  git commit -m "Résultats et rapport du TP déploiement IA"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
else
  git push -u origin main
fi

echo "Dépôt publié : $(gh repo view --json url -q .url)"
