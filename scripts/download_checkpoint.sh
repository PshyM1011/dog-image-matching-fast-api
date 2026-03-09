```bash
#!/bin/bash
set -e
if [ -z "$CHECKPOINT_URL" ]; then
  echo "CHECKPOINT_URL not set; skipping checkpoint download"
  exit 0
fi
echo "Downloading checkpoint from $CHECKPOINT_URL ..."
curl -fSL -o best.pth "$CHECKPOINT_URL"
echo "Done. best.pth is in project root."
```