from collections import OrderedDict
from typing import Any, Dict, Dict as OrderedDictType

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from robustbench_ghost.model_zoo.imagenet import imagenet_models
from robustbench_ghost.model_zoo.enums import BenchmarkDataset, ThreatModel

ModelsDict = OrderedDictType[str, Dict[str, Any]]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.imagenet, imagenet_models)
])
