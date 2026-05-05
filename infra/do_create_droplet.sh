#!/bin/bash
set -e

# Better AI - DigitalOcean Droplet Creation Script
# This script uses 'doctl' to create a GPU Droplet based on infra/droplet_config.yml

echo "🏗️ Creating Better AI GPU Droplet..."

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

echo "Configured Settings:"
echo "- Size: $SIZE"
echo "- Region: $REGION"
echo "- Image: $IMAGE"
echo "- Tags: $TAGS"

# Provisioning script
USER_DATA_FILE="infra/do_provision.sh"

# Create the Droplet
echo "🚀 Dispatching droplet creation command..."
doctl compute droplet create better-ai-gpu \
    --region "$REGION" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --user-data-file "$USER_DATA_FILE" \
    --tag-names "$TAGS" \
    --ssh-keys "$SSH_KEYS" \
    --wait

echo "✅ Droplet created successfully!"
doctl compute droplet list --tag-name better-ai
