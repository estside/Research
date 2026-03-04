import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Testing pandas...")
import pandas as pd
print("Testing torch...")
import torch
print("Testing re...")
import re
print("Testing Bio...")
from Bio import SeqIO
print("Testing transformers...")
from transformers import AutoTokenizer, EsmModel
print("All imports successful!")
