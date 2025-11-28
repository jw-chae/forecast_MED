from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

@dataclass
class APISettings:
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # Future API settings can be added here

@dataclass
class SEIRParams:
    population: int = 10_000_000  # Default for Hangzhou
    initial_exposed: int = 10
    initial_recovered: int = 0
    infectious_days: float = 7.0
    incubation_days: float = 5.0
    # beta and incubation_days are proposed by the LLM

@dataclass
class LocationSettings:
    # Key: location name, Value: (latitude, longitude)
    locations: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "hangzhou": (30.2741, 120.1551)
    })

@dataclass
class SimulationDefaults:
    # Corresponds to argparse defaults in rolling_agent_forecast.py
    disease: str = "手足口病"
    start: str = "2024-04-01"
    end: str = "2024-06-30"
    n_steps: int = 4
    horizon: int = 8
    provider: str = "dashscope"
    model: str = "Qwen/Qwen3-235B-A22B-Thinking-2507"
    temperature: float = 0.2
    train_len_weeks: Optional[int] = None
    no_llm: bool = False
    save_json: bool = False

@dataclass
class AppConfig:
    api: APISettings = field(default_factory=APISettings)
    seir: SEIRParams = field(default_factory=SEIRParams)
    locations: LocationSettings = field(default_factory=LocationSettings)
    simulation: SimulationDefaults = field(default_factory=SimulationDefaults)

# Singleton instance for the application settings
APP_CONFIG = AppConfig()
