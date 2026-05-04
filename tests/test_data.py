from datasets import load_dataset_builder

builder = load_dataset_builder("agkphysics/AudioSet", "balanced")

print(builder.info)