# Releasing kindling

Releases are cut by pushing a `v*` tag. GitHub Actions builds per-platform
abi3 wheels (Linux x86_64, macOS arm64/x86_64, Windows x86_64) + an sdist,
then publishes to PyPI via **Trusted Publishing** (OIDC — no stored token).

## One-time PyPI setup (do this once before the first release)

1. Log in to [pypi.org](https://pypi.org) (create an account if needed).
2. Go to **Your projects → Add project** (or the project page once it exists).
3. Under **Publishing → Trusted Publishers → Add a new publisher**, fill in:
   - **PyPI project name:** `kindling`
   - **Owner:** `rhoekstr`
   - **Repository:** `kindling`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`
4. Save. No API token is needed after this.

## Cutting a release

```bash
# 1. Make sure pyproject.toml and native/kindling_core/Cargo.toml both carry
#    the new version (e.g. 1.0.0). Commit + merge to master.

# 2. Push a matching tag:
git tag v1.0.0
git push origin v1.0.0
```

The `release.yml` workflow triggers automatically, builds all wheels, runs a
smoke-test on each, and publishes the full dist set to PyPI.

## Verifying the release

```bash
pip install kindling==1.0.0
python -c "import kindling; print(kindling.__version__)"
```

## Version bump checklist

- `pyproject.toml` → `version = "X.Y.Z"`
- `native/kindling_core/Cargo.toml` → `version = "X.Y.Z"`
- Git tag → `vX.Y.Z`
