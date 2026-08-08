# Sync the package to Databricks

## Download and upload audio MP3s

`MediaStore` reads from `/Volumes/luka_erceg/drzavni_posao/audios`.
The episode utility uploads existing local MP3s, then downloads and immediately
uploads missing episodes:

```sh
databricks auth describe --profile staging
uv run episodes.py --dry-run
uv run episodes.py
```

The command processes season 3 by default. Use comma-separated `--season` or
`--episode` values to select specific media. `--downloads`, `--volume`,
`--profile`, and `--overwrite` customize the local and remote locations.
Add `--local` to download without accessing Databricks.

## Sync the Python package

Authenticate with the `staging` profile, then run from the repository root:

```sh
databricks auth describe --profile staging
databricks sync drzavni_posao \
  /Users/luka.erceg@databricks.com/drzavni_posao/drzavni_posao \
  --profile staging \
  --full \
  --exclude "__pycache__/**" \
  --exclude "*.pyc"
```

Add `--dry-run` to preview the upload. This syncs only the Python package; it
does not upload notebooks, tests, gallery data, or other repository files.

## Sync notebooks

Sync every notebook in one command:

```sh
databricks sync notebooks \
  /Users/luka.erceg@databricks.com/drzavni_posao \
  --profile staging \
  --full
```

Add `--dry-run` to preview the notebook sync. The notebooks stay at the
workspace root so their existing package imports continue to work.
