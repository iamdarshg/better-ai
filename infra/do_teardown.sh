#!/bin/bash

# Better AI - DigitalOcean Infrastructure Teardown Script
# This script automates the decommissioning of training infrastructure using 'doctl'.

echo "🧹 Better AI Infrastructure Teardown"
echo "=================================================="

if ! command -v doctl &> /dev/null; then
    echo "❌ Error: 'doctl' is not installed. Please follow the manual checklist in this script."
    # Fallback to manual checklist printing
    cat <<EOF
⚠️  WARNING: This will stop all training and can lead to data loss if not careful.

Recommended Manual Cleanup Checklist:
--------------------------------------------------
[ ] 📸 Snapshot the checkpoint volume (if you want to keep the trained model)
[ ] 🗑️  Destroy the Droplet (to stop incurring hourly costs)
[ ] 🗑️  Detach and Destroy the block storage volume (if no longer needed)

Visit https://cloud.digitalocean.com/droplets to manually destroy your Droplet.
EOF
    exit 0
fi

# Automated Teardown with doctl
DROPLET_ID=$(doctl compute droplet list --tag-name better-ai --format ID --no-header | head -n 1)

if [ -z "$DROPLET_ID" ]; then
    echo "ℹ️ No droplets found with tag 'better-ai'."
    exit 0
fi

echo "Detected Droplet ID: $DROPLET_ID"
read -p "Do you want to snapshot the volumes and destroy this Droplet? (y/N) " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Teardown cancelled."
    exit 0
fi

# 1. Snapshot volumes
echo "📸 Checking for attached volumes..."
VOLUME_IDS=$(doctl compute droplet get "$DROPLET_ID" --format Volumes --no-header)

if [ -n "$VOLUME_IDS" ] && [ "$VOLUME_IDS" != "[]" ]; then
    for VOL_ID in $(echo $VOLUME_IDS | tr ',' ' '); do
        echo "Creating snapshot for volume $VOL_ID..."
        TIMESTAMP=$(date +%Y%m%d%H%M%S)
        doctl compute volume snapshot "$VOL_ID" --snapshot-name "better-ai-snap-$TIMESTAMP"
    done
fi

# 2. Destroy Droplet
echo "🗑️ Destroying Droplet $DROPLET_ID..."
doctl compute droplet delete "$DROPLET_ID" --force

echo "✅ Teardown initiated."
echo "💡 Cost Tip: Ensure any detached volumes are also deleted if no longer needed to stop all billing."
echo "=================================================="
