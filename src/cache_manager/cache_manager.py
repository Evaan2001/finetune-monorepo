import os
import sys
import hashlib
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    # Try the local import path first
    from src.training.train_model import train_classifier
except ModuleNotFoundError:
    # Fall back to the remote import path
    from ..training.train_model import train_classifier


class ModelCacheManager:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        print("-" * 80)
        print("Cache directory Set To:", self.cache_dir)
        print("-" * 80)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_path(self, model_name, revision="main"):
        """Get the path where a model should be cached"""
        # Create a unique identifier for this model version
        model_id = f"{model_name}_{revision}"
        cache_hash = hashlib.sha256(model_id.encode()).hexdigest()

        return Path(self.cache_dir) / cache_hash

    def is_cached_manually(self, model_name, revision="main"):
        """
        Check if a model was cached manually (so not Hugging Face)
        By checking the existence of cached files saved under the cache directory
        """

        # Determine the cache path for this model
        model_cache_path = self.get_cache_path(model_name, revision)
        return model_cache_path.exists()

    def is_cached_by_hugging_face(self, model_name, revision="main"):
        """
        Check if a model is already cached automatically by Hugging Face
        By checking the existence of cached files saved under Hugging Face's
        default naming conventions
        """

        # Determine Hugging Face's cache path
        hf_cache_dir = (
            self.cache_dir
            if "huggingface/hub" in self.cache_dir
            else "~/.cache/huggingface/hub"
        )
        hf_cache_dir = os.path.expanduser(hf_cache_dir)  # Expand ~ to home directory
        # Convert model_name format: "org/model" → "models--org--model"
        hf_model_path = "models--" + model_name.replace("/", "--")
        hugging_face_cache_path = Path(hf_cache_dir) / hf_model_path
        # Simple directory existence check
        return hugging_face_cache_path.exists()

    # We don't have a save_model as the train_model function will save the model

    def load_model(self, model_name, revision="main"):
        model_path = self.get_cache_path(model_name, revision)
        if model_path.exists():
            print(f"Loading model from {model_path}")
            # Load the model from the cache
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        else:
            print(f"Model not found in cache at {model_path}")
            return None


if __name__ == "__main__":
    # Packages needed just for demoing and not for the cache manager itself
    import time
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Choose a very lightweight model for demonstration
    model_name = "prajjwal1/bert-tiny"  # Only about 17MB, 2-layer BERT
    revision = "main"

    # Initialize the cache manager
    cache_manager = ModelCacheManager()

    print("=" * 100)
    print(
        "FIRST RUN WITH A HUGGING FACE MODEL - MODEL SHOULD NOT BE IN OUR CACHE UNLESS USED PREVIOUSLY IN YOUR ENVIRONMENT"
    )
    print("=" * 100)

    print(
        "Is model cached by Hugging Face: ",
        cache_manager.is_cached_by_hugging_face(model_name, revision),
    )
    start_time = time.time()
    # First run - should download the model
    tokenize1 = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model1 = AutoModel.from_pretrained(model_name, revision=revision)
    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds")

    print("\n" + "=" * 80)
    print("SECOND RUN WITH A HUGGING FACE MODEL - MODEL SHOULD BE IN OUR CACHE")
    print("=" * 80)

    print(
        "Is model cached by Hugging Face: ",
        cache_manager.is_cached_by_hugging_face(model_name, revision),
    )

    start_time = time.time()
    # Second run - should use cached model
    tokenizer2 = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model2 = AutoModel.from_pretrained(model_name, revision=revision)
    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds\n")

    # Check if models are the same
    print("=" * 80)
    print("Comparing model parameters...")
    print("=" * 80)
    for (name1, p1), (name2, p2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if not torch.allclose(p1, p2):
            print(f"Parameters differ: {name1}")
            break
    else:
        print(
            "All parameters match between runs - successfully using the same model!\n"
        )

    print("*" * 80)
    print("FIRST RUN WITH A MANUALLY TRAINED MODEL")
    print("*" * 80)
    model_name = "prajjwal1/bert-tiny"
    revision = "main"
    model_path = cache_manager.get_cache_path(model_name, revision)

    # First run - should train the model
    is_model_available = cache_manager.is_cached_manually(model_name, revision)
    print("Is a manually trained model cached: ", is_model_available)
    start_time = time.time()
    if not is_model_available:
        print("Training a new model...")
        # Train the model and save it to the cache
        output_path = train_classifier(
            model_name=model_name,
            output_dir=str(model_path),
        )
        # Now load the model
        model_1, tokenizer_1 = cache_manager.load_model(model_name, revision)
    else:
        print("Loading model from cache...")
        # Load the model from the cache
        model_1, tokenizer_1 = cache_manager.load_model(model_name, revision)

    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds")

    print("\n" + "*" * 80)
    print("SECOND RUN WITH A MANUALLY TRAINED MODEL - MODEL SHOULD BE IN OUR CACHE")
    print("*" * 80)

    # Second run - should use cached model
    is_model_available = cache_manager.is_cached_manually(model_name, revision)
    print("Is a manually trained model cached: ", is_model_available)
    start_time = time.time()
    if not is_model_available:
        print("Training a new model...")
        # Train the model and save it to the cache
        output_path = train_classifier(
            model_name=model_name,
            output_dir=str(model_path),
        )
        # Now load the model
        model_2, tokenizer_2 = cache_manager.load_model(model_name, revision)
    else:
        print("Loading model from cache...")
        # Load the model from the cache
        model_2, tokenizer_2 = cache_manager.load_model(model_name, revision)

    end_time = time.time()
    print(f"Model loading time: {end_time - start_time:.2f} seconds\n")

    # Check if models are the same
    print("*" * 80)
    print("Comparing model parameters...")
    print("*" * 80)
    for (name1, p1), (name2, p2) in zip(
        model_1.named_parameters(), model_2.named_parameters()
    ):
        if not torch.allclose(p1, p2):
            print(f"Parameters differ: {name1}")
            break
    else:
        print("All parameters match between runs - successfully using the same model!")
