#!/bin/bash

# Better AI - DigitalOcean Infrastructure Teardown Guide
# This script provides commands and instructions to safely teardown training infrastructure.

echo "🧹 Better AI Infrastructure Teardown"
echo "=================================================="

echo "⚠️  WARNING: This will stop all training and can lead to data loss if not careful."
echo ""

# 1. Stop active containers
echo "Step 1: Stopping Docker containers..."
if command -v docker &> /dev/null && [ -f "docker-compose.yml" ]; then
    docker compose down
else
    echo "Command: docker compose down"
fi
echo ""

# 2. Checklist for manual teardown via doctl or DO Cloud Console
echo "Step 2: Recommended Manual Cleanup Checklist:"
echo "--------------------------------------------------"
echo "[ ] 📸 Snapshot the checkpoint volume (if you want to keep the trained model)"
echo "[ ] 🗑️  Destroy the Droplet (to stop incurring hourly costs)"
echo "[ ] 🗑️  Detach and Destroy the block storage volume (if no longer needed)"
echo "[ ] 🗑️  Remove any unused Snapshots (to save on storage costs)"
echo ""

# 3. Helper commands if 'doctl' is installed
if command -v doctl &> /dev/null; then
    echo "Step 3: doctl helper commands:"
    echo "--------------------------------------------------"
    echo "# List droplets to find ID"
    DROPLET_ID=$(doctl compute droplet list --tag-name better-ai --format ID --no-header | head -n 1)

    if [ -n "$DROPLET_ID" ]; then
        echo "Detected Droplet ID: $DROPLET_ID"
        echo ""
        echo "To destroy this droplet:"
        echo "doctl compute droplet delete $DROPLET_ID"
    else
        echo "doctl compute droplet list --tag-name better-ai"
        echo ""
        echo "# Destroy droplet (REPLACE <ID> with your droplet ID)"
        echo "doctl compute droplet delete <ID>"
    fi
else
    echo "Step 3: Visit https://cloud.digitalocean.com/droplets to manually destroy your Droplet."
fi

echo ""
echo "💡 Cost Tip: H100 Droplets cost ~$10-15/hour. Always destroy the Droplet when training is finished!"
echo "=================================================="
