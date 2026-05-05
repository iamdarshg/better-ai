#!/bin/bash
set -e

# Better AI - DigitalOcean Droplet Creation Script
# This script uses 'doctl' to create a GPU Droplet and a Block Storage Volume.

echo "🏗️ Creating Better AI Infrastructure..."

if ! command -v doctl &> /dev/null; then
    echo "❌ Error: 'doctl' is not installed. Please install it and authenticate first."
    exit 1
fi

# Simple parser for the YAML config (handles basic key-value pairs)
parse_yaml() {
    local prefix=$2
    local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
    sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
    awk -F$fs '{
        indent = length($1)/2;
        vname[indent] = $2;
        for (i in vname) {if (i > indent) {delete vname[i]}}
        if (length($3) > 0) {
            vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
            printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
        }
    }'
}

# Load configuration
CONFIG_FILE="infra/droplet_config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found."
    exit 1
fi

# Extract values from config
eval $(parse_yaml "$CONFIG_FILE" "CONF_")

SIZE=${CONF_droplet_size:-"gpu-h100x1-80gb"}
REGION=${CONF_droplet_region:-"nyc3"}
IMAGE=${CONF_droplet_image:-"ubuntu-22-04-x64"}
TAGS=${CONF_droplet_tags:-"better-ai"}
SSH_KEYS=${CONF_networking_ssh_keys:-""}

VOL_NAME=${CONF_storage_volume_name:-""}
VOL_SIZE=${CONF_storage_volume_size_gb:-"100"}

echo "Configured Settings:"
echo "- Size: $SIZE"
echo "- Region: $REGION"
echo "- Image: $IMAGE"
echo "- Tags: $TAGS"

# 1. Create Block Storage Volume if specified
if [ -n "$VOL_NAME" ]; then
    echo "💾 Creating/Checking Block Storage Volume: $VOL_NAME..."
    if ! doctl compute volume get "$VOL_NAME" --region "$REGION" > /dev/null 2>&1; then
        doctl compute volume create "$VOL_NAME" --region "$REGION" --size "${VOL_SIZE}GiB" --desc "Better AI Checkpoints"
        echo "✅ Volume created."
    else
        echo "ℹ️ Volume already exists."
    fi
fi

# 2. Create the Droplet
echo "🚀 Dispatching droplet creation command..."
# Create droplet and capture its ID
DROPLET_ID=$(doctl compute droplet create better-ai-gpu \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --user-data-file "infra/do_provision.sh" \
    --tag-names "$TAGS" \
    --ssh-keys "$SSH_KEYS" \
    --format ID --no-header \
    --wait)

echo "✅ Droplet created with ID: $DROPLET_ID"

# 3. Attach Volume if specified
if [ -n "$VOL_NAME" ] && [ -n "$DROPLET_ID" ]; then
    echo "🔗 Attaching volume $VOL_NAME to droplet $DROPLET_ID..."
    doctl compute volume-action attach "$VOL_NAME" "$DROPLET_ID"
    echo "✅ Volume attached."
fi

echo "Environment is provisioning. Monitor progress by SSHing into the droplet."
doctl compute droplet list --tag-name better-ai
