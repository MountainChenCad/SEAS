"""
Unified Configuration System for TableLlama-HRRP

Single source of truth for:
- Default values (previously scattered in config.py)
- Configuration dataclasses (previously in eval_config.py)
- Configuration management (previously in config_manager.py)

This module consolidates 3 separate config files into one organized system.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional


# ============================================================================
# 1. DATASET CONFIGURATION
# ============================================================================

DEFAULT_DATASET_KEY = "simulated"
AVAILABLE_DATASETS = {
    "simulated": {
        "path": "datasets/simulated_hrrp",
        "data_var": "CoHH",
        "original_len": 1000,
        "max_samples_to_load": None
    },
    "measured": {
        "path": "datasets/measured_hrrp",
        "data_var": "hrrp",
        "original_len": 500,
        "max_samples_to_load": None
    },
}
TARGET_HRRP_LENGTH = 1000
PREPROCESS_MAT_TO_NPY = True
PROCESSED_DATA_DIR = "data_processed"
TEST_SPLIT_SIZE = 0.3
RANDOM_STATE = 42


# ============================================================================
# 2. SCATTERING CENTER EXTRACTION CONFIGURATION
# ============================================================================

SCATTERING_CENTER_EXTRACTION = {
    "enabled": True,
    "method": "peak_detection",
    "prominence": 0.15,
    "min_distance": 5,
    "max_centers_to_keep": 10,
    "normalize_hrrp_before_extraction": True,
    "normalization_type_for_hrrp": "max"
}

SCATTERING_CENTER_ENCODING = {
    "format": "list_of_dicts",
    "precision_pos": 0,
    "precision_amp": 3,
    "center_separator": "; ",
    "pos_amp_separator": ":",
    "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH
}


# ============================================================================
# 3. FSL TASK SETUP CONFIGURATION
# ============================================================================

DEFAULT_FSL_TASK_SETUP = {
    "enabled": True,
    "n_way": 5,
    "k_shot_support": 1,
    "q_shot_query": 1,
    "num_fsl_tasks": 20,
    "sc_feature_type_for_prototype": "pos_amp_flat"
}


# ============================================================================
# 4. LLM API & PROMPT CONFIGURATION
# ============================================================================

DEFAULT_LLM_CALLER_PARAMS = {
    "temperature": 0.1,
    "top_p": 1.0,
    "max_tokens_completion": 250,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "api_retry_delay": 10,
    "max_retries": 5
}
DEFAULT_NUM_CONSISTENCY_PATHS = 1
DEFAULT_CONSISTENCY_TEMPERATURE = 1.0

DEFAULT_API_KEY = "local"
DEFAULT_BASE_URL = "YOUR_DEFAULT_BASE_URL_HERE"
DEFAULT_API_PROVIDER = "qwen_local"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen"

DEFAULT_LIMIT_TEST_SAMPLES = None


# ============================================================================
# 5. RESULTS AND OUTPUT DIRECTORIES
# ============================================================================

# Primary results directory (actively used)
RESULTS_BASE_DIR = "results"

# Future expansion points (reserved but not currently used)
# Uncomment and use these if you plan to:
# - Run ablation studies and store results separately
# - Generate LaTeX tables for papers
RESULTS_ABLATION_DIR = "results_ablation"      # Reserved for ablation study results
GENERATED_TABLES_DIR = "paper_tables"          # Reserved for generated LaTeX tables


# ============================================================================
# 6. BASELINE CONFIGURATION
# ============================================================================

RUN_BASELINE_SVM = True
BASELINE_SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "feature_type": "scattering_centers",
    "sc_feature_type_for_svm": "pos_amp_flat"
}

RUN_BASELINE_RF = True
BASELINE_RF_PARAMS = {
    "n_estimators": 100,
    "feature_type": "scattering_centers",
    "sc_feature_type_for_rf": "pos_amp_flat"
}


# ============================================================================
# 7. CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class DatasetConfig:
    """Dataset loading configuration"""
    processed_data_dir: str = "data_processed"
    dataset_key: str = "simulated"
    random_state: int = 42


@dataclass
class TrainingConfig:
    """LoRA training configuration - 统一使用论文标准配置 r=2, alpha=32"""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    grad_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 100
    lora_rank: int = 2         # 统一为2，论文标准
    lora_alpha: int = 32       # alpha=16*r
    lora_dropout: float = 0.1  # 论文标准
    max_length: int = 5000
    precision: str = "fp16"


@dataclass
class ValidationConfig:
    """Validation and model selection configuration"""
    eval_loss_weight: float = 0.5
    eval_fsl_weight: float = 0.5
    save_best_metric: str = "combined"  # combined, loss, or fsl_accuracy
    fsl_n_way: int = 3
    fsl_k_shot: int = 1
    fsl_q_shot: int = 1
    fsl_num_tasks: int = 20


@dataclass
class DataSplitConfig:
    """Data splitting configuration for train/val/test"""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42


@dataclass
class ScatteringCenterConfig:
    """Scattering center extraction and encoding configuration"""
    # Extraction config
    extraction_enabled: bool = True
    max_centers_to_keep: int = 10
    prominence: float = 0.15
    min_distance: int = 5
    normalize_hrrp_before_extraction: bool = True
    normalization_type: str = "max"

    # Encoding config
    encoding_format: str = "list_of_dicts"
    precision_pos: int = 0
    precision_amp: int = 3
    hrrp_length: int = 1000
    center_separator: str = "; "
    pos_amp_separator: str = ":"

    def to_extraction_dict(self) -> dict:
        """Convert to format expected by data_utils"""
        return {
            "enabled": self.extraction_enabled,
            "max_centers_to_keep": self.max_centers_to_keep,
            "prominence": self.prominence,
            "min_distance": self.min_distance,
            "normalize_hrrp_before_extraction": self.normalize_hrrp_before_extraction,
            "normalization_type_for_hrrp": self.normalization_type,
        }

    def to_encoding_dict(self) -> dict:
        """Convert to format expected by scattering_center_encoder"""
        return {
            "format": self.encoding_format,
            "precision_pos": self.precision_pos,
            "precision_amp": self.precision_amp,
            "TARGET_HRRP_LENGTH_INFO": self.hrrp_length,
            "center_separator": self.center_separator,
            "pos_amp_separator": self.pos_amp_separator,
        }


@dataclass
class FSLConfig:
    """Few-Shot Learning task setup configuration"""
    n_way: int = 5
    k_shot_support: int = 1
    q_shot_query: int = 1
    num_fsl_tasks: int = 20
    sc_feature_type_for_prototype: str = "pos_amp_flat"


@dataclass
class LLMConfig:
    """LLM API and generation configuration"""
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens_completion: int = 250
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_retry_delay: int = 10
    max_retries: int = 5
    api_provider: str = "qwen_local"
    model_path: str = "/root/autodl-tmp/Qwen"
    api_key: str = "local"


@dataclass
class SiliconFlowConfig:
    """SiliconFlow API configuration for fine-tuned models"""
    api_key: str = ""  # 从环境变量 SILICONFLOW_API_KEY 读取
    base_url: str = "https://api.siliconflow.cn/v1"

    # 三个微调模型的标识符
    model_initial: str = "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj"
    model_ckpt_406: str = "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_406"
    model_ckpt_203: str = "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:rpl47v9x40:initial_commit:uyulemtufwhthcnywhcj-ckpt_step_203"

    # 推理超参数
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 3000

    # API调用配置
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0  # 指数退避的基础延迟


# ============================================================================
# 8. DEFAULT SINGLETON INSTANCES
# ============================================================================

DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_VALIDATION_CONFIG = ValidationConfig()
DEFAULT_DATA_SPLIT_CONFIG = DataSplitConfig()
DEFAULT_SC_CONFIG = ScatteringCenterConfig()
DEFAULT_FSL_CONFIG = FSLConfig()
DEFAULT_LLM_CONFIG = LLMConfig()


# ============================================================================
# 9. BACKWARD COMPATIBILITY - SimpleConfig
# ============================================================================

class SimpleConfig:
    """
    Backward-compatible config class (previously scattered in eval scripts).
    Now uses unified defaults from config.py
    """
    def __init__(
        self,
        sc_config: ScatteringCenterConfig = None,
        data_config: DatasetConfig = None
    ):
        self.sc_config = sc_config or DEFAULT_SC_CONFIG
        self.data_config = data_config or DEFAULT_DATASET_CONFIG

        # These attributes are expected by data_utils and scattering_center_encoder
        self.PROCESSED_DATA_DIR = self.data_config.processed_data_dir
        self.RANDOM_STATE = self.data_config.random_state
        self.sc_extraction_config = self.sc_config.to_extraction_dict()
        self.sc_encoding_config = self.sc_config.to_encoding_dict()


# Default singleton instance for backward compatibility
eval_config = SimpleConfig()


# ============================================================================
# 10. CONFIGURATION MANAGER (FACTORY PATTERN)
# ============================================================================

class ConfigManager:
    """
    Central configuration manager.
    Unified factory for loading configuration from multiple sources:
    - Defaults
    - CLI arguments
    - YAML/JSON files
    - Environment variables
    """

    @staticmethod
    def from_defaults() -> Dict[str, Any]:
        """Load default configuration from defaults"""
        return {
            "dataset_key": DEFAULT_DATASET_KEY,
            "n_way": DEFAULT_FSL_TASK_SETUP.get("n_way", 5),
            "k_shot_support": DEFAULT_FSL_TASK_SETUP.get("k_shot_support", 1),
            "q_shot_query": DEFAULT_FSL_TASK_SETUP.get("q_shot_query", 1),
            "num_fsl_tasks": DEFAULT_FSL_TASK_SETUP.get("num_fsl_tasks", 20),
            "temperature": DEFAULT_LLM_CALLER_PARAMS.get("temperature", 0.1),
            "top_p": DEFAULT_LLM_CALLER_PARAMS.get("top_p", 1.0),
            "max_tokens_completion": DEFAULT_LLM_CALLER_PARAMS.get("max_tokens_completion", 250),
            "api_provider": DEFAULT_API_PROVIDER,
            "model_path": DEFAULT_MODEL_PATH,
            "results_dir": RESULTS_BASE_DIR,
            "processed_data_dir": PROCESSED_DATA_DIR,
            "random_state": RANDOM_STATE,
        }

    @staticmethod
    def from_cli(args: argparse.Namespace) -> Dict[str, Any]:
        """Load configuration from CLI arguments and merge with defaults"""
        config = ConfigManager.from_defaults()
        cli_args = vars(args)
        for key, value in cli_args.items():
            if value is not None:
                config[key] = value
        return config

    @staticmethod
    def from_yaml(yaml_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            return ConfigManager._merge_with_defaults(yaml_config)
        except ImportError:
            print("Warning: PyYAML not installed. Cannot load YAML config.")
            return ConfigManager.from_defaults()
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

    @staticmethod
    def from_json(json_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        return ConfigManager._merge_with_defaults(json_config)

    @staticmethod
    def _merge_with_defaults(custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom config with defaults"""
        defaults = ConfigManager.from_defaults()
        defaults.update(custom_config)
        return defaults

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """Validate configuration for consistency and required fields"""
        required_fields = ["dataset_key", "n_way", "num_fsl_tasks"]
        for field in required_fields:
            if field not in config or config[field] is None:
                raise ValueError(f"Missing required config field: {field}")

        if config["n_way"] <= 0:
            raise ValueError("n_way must be positive")
        if config["num_fsl_tasks"] <= 0:
            raise ValueError("num_fsl_tasks must be positive")
        if not (0 < config["temperature"] <= 2.0):
            raise ValueError("temperature should be between 0 and 2.0")

        if config["dataset_key"] not in AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {config['dataset_key']}")

        return True

    @staticmethod
    def get_data_config(config: Dict[str, Any]) -> DatasetConfig:
        """Extract data configuration as DatasetConfig object"""
        return DatasetConfig(
            processed_data_dir=config.get("processed_data_dir", PROCESSED_DATA_DIR),
            dataset_key=config.get("dataset_key", DEFAULT_DATASET_KEY),
            random_state=config.get("random_state", RANDOM_STATE),
        )

    @staticmethod
    def get_sc_config(config: Dict[str, Any]) -> ScatteringCenterConfig:
        """Extract scattering center configuration"""
        return ScatteringCenterConfig(
            extraction_enabled=config.get("extraction_enabled", True),
            max_centers_to_keep=config.get("max_centers_to_keep", 10),
            prominence=config.get("prominence", SCATTERING_CENTER_EXTRACTION.get("prominence", 0.15)),
            min_distance=config.get("min_distance", 5),
            normalize_hrrp_before_extraction=config.get("normalize_hrrp_before_extraction", True),
            normalization_type=config.get("normalization_type_for_hrrp", "max"),
            hrrp_length=config.get("hrrp_length", TARGET_HRRP_LENGTH),
        )

    @staticmethod
    def get_sc_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scattering center extraction configuration (legacy dict format)"""
        return {
            "enabled": True,
            "max_centers_to_keep": config.get("max_centers_to_keep", 10),
            "prominence": config.get("prominence", SCATTERING_CENTER_EXTRACTION.get("prominence", 0.15)),
            "min_distance": config.get("min_distance", 5),
        }

    @staticmethod
    def get_sc_encoding_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scattering center encoding configuration (legacy dict format)"""
        return {
            "format": "list_of_dicts",
            "precision_pos": 0,
            "precision_amp": 3,
            "TARGET_HRRP_LENGTH_INFO": TARGET_HRRP_LENGTH,
        }

    @staticmethod
    def get_fsl_config(config: Dict[str, Any]) -> FSLConfig:
        """Extract FSL task configuration"""
        return FSLConfig(
            n_way=config.get("n_way", DEFAULT_FSL_TASK_SETUP.get("n_way", 5)),
            k_shot_support=config.get("k_shot_support", DEFAULT_FSL_TASK_SETUP.get("k_shot_support", 1)),
            q_shot_query=config.get("q_shot_query", DEFAULT_FSL_TASK_SETUP.get("q_shot_query", 1)),
            num_fsl_tasks=config.get("num_fsl_tasks", DEFAULT_FSL_TASK_SETUP.get("num_fsl_tasks", 20)),
        )

    @staticmethod
    def get_llm_config(config: Dict[str, Any]) -> LLMConfig:
        """Extract LLM caller configuration"""
        return LLMConfig(
            temperature=config.get("temperature", DEFAULT_LLM_CALLER_PARAMS.get("temperature", 0.1)),
            top_p=config.get("top_p", DEFAULT_LLM_CALLER_PARAMS.get("top_p", 1.0)),
            max_tokens_completion=config.get("max_tokens_completion", DEFAULT_LLM_CALLER_PARAMS.get("max_tokens_completion", 250)),
            frequency_penalty=config.get("frequency_penalty", DEFAULT_LLM_CALLER_PARAMS.get("frequency_penalty", 0.0)),
            presence_penalty=config.get("presence_penalty", DEFAULT_LLM_CALLER_PARAMS.get("presence_penalty", 0.0)),
            api_retry_delay=config.get("api_retry_delay", DEFAULT_LLM_CALLER_PARAMS.get("api_retry_delay", 10)),
            max_retries=config.get("max_retries", DEFAULT_LLM_CALLER_PARAMS.get("max_retries", 5)),
            api_provider=config.get("api_provider", DEFAULT_API_PROVIDER),
            model_path=config.get("model_path", DEFAULT_MODEL_PATH),
            api_key=config.get("api_key", DEFAULT_API_KEY),
        )

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """Save configuration to file for reproducibility"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {output_path}")


# ============================================================================
# 11. DPO TRAINING CONFIGURATION
# ============================================================================

@dataclass
class DPOTrainingConfig:
    """DPO (Direct Preference Optimization) 训练配置"""
    num_epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 5e-5
    beta: float = 0.1  # DPO温度参数
    max_length: int = 5000
    max_prompt_length: int = 2500
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    save_steps: int = 100
    logging_steps: int = 10
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid or hinge

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# DPO训练默认参数
DEFAULT_DPO_TRAINING_PARAMS = {
    "num_epochs": 2,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "beta": 0.1,
    "max_length": 5000,
    "lora_rank": 16,
    "lora_alpha": 32,
    "gradient_accumulation_steps": 4,
}

# DPO数据集路径
DPO_DATA_PATH = "data/hrrp_dpo_train.json"
DPO_STATS_PATH = "data/hrrp_dpo_stats.json"
DPO_OUTPUT_DIR = "output/qwen3-hrrp-dpo-baseline"
