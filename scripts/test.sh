#!/usr/bin/env bash
set -euo pipefail

echo "==> Formatting check"
cargo fmt --all -- --check

echo "==> Compile check"
cargo check --workspace --locked

echo "==> Running tests"
cargo test --workspace --locked
