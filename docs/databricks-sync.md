# Sync the package to Databricks

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

Upload the notebooks separately:

```sh
databricks workspace import \
  /Users/luka.erceg@databricks.com/drzavni_posao/pipeline \
  --profile staging --format JUPYTER --language PYTHON --overwrite \
  --file pipeline.ipynb

databricks workspace import \
  /Users/luka.erceg@databricks.com/drzavni_posao/label_characters \
  --profile staging --format JUPYTER --language PYTHON --overwrite \
  --file label_characters.ipynb

databricks workspace import \
  /Users/luka.erceg@databricks.com/drzavni_posao/large-v3-hotwords-assignment-nb \
  --profile staging --format JUPYTER --language PYTHON --overwrite \
  --file large-v3-hotwords-assignment-nb.ipynb
```
