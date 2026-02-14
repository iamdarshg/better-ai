
import yaml
import logging
from better_ai.data.unified_dataloader import StreamingDataset
from transformers import AutoTokenizer
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_datasets(config_path="datasets.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", [])
    if not datasets:
        logger.error("No datasets found in config")
        return False

    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")

    success_count = 0
    failure_count = 0

    for ds_config in datasets:
        name = ds_config.get("name")
        path = ds_config.get("path")
        config_name = ds_config.get("config_name")
        split = ds_config.get("split", "train")

        logger.info(f"Validating dataset: {name} ({path})")

        try:
            # We use a very short timeout or just check if it can be initialized
            # Streaming=True should be fast as it doesn't download everything
            ds = StreamingDataset(
                dataset_name=path,
                config_name=config_name,
                tokenizer=tokenizer,
                split=split,
                streaming=True
            )
            logger.info(f"Successfully initialized {name}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            failure_count += 1

    logger.info(f"Validation complete: {success_count} success, {failure_count} failures")
    return failure_count == 0

if __name__ == "__main__":
    if not validate_datasets():
        sys.exit(1)
