import pytest
import numpy as np
import gymnasium as gym
from src.environment.dynamic_pricing_env import DynamicPricingEnv

@pytest.fixture
def env():
    return DynamicPricingEnv(max_inventory=100, max_days=30, base_arrivals=15)

def test_environment_initialization(env):
    # Observation space check
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (5,)
    
    # Action space check
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 5
    
    # Pricing level list matches action space n
    assert len(env.pricing_levels) == 5

def test_environment_reset(env):
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (5,)
    assert isinstance(obs, np.ndarray)
    assert obs[0] == 100.0  # max_inventory
    assert obs[1] == 30.0   # max_days
    assert 90.0 <= obs[2] <= 140.0 # competitor price
    assert obs[3] in [1.0, 1.2, 1.5, 1.8] # seasonality factor
    assert obs[4] in [0, 1, 2, 3] # market demand level
    
    assert isinstance(info, dict)
    assert "cumulative_revenue" in info
    assert info["cumulative_revenue"] == 0.0
    assert "seasonality_label" in info

def test_environment_step(env):
    obs, info = env.reset(seed=42)
    initial_inventory = obs[0]
    initial_days = obs[1]
    
    # Take action 2 (pricing level 120)
    next_obs, reward, terminated, truncated, info = env.step(2)
    
    # Assert return types
    assert next_obs.shape == (5,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Assert values
    assert next_obs[0] <= initial_inventory  # inventory should either decrease or stay the same
    assert next_obs[1] == initial_days - 1   # days should decrement by 1
    assert reward == info["tickets_sold"] * 120.0
    assert env.cumulative_revenue == reward
    
    assert "tickets_sold" in info
    assert "selected_price" in info
    assert info["selected_price"] == 120
    assert "daily_revenue" in info
    assert "cumulative_revenue" in info

def test_invalid_action(env):
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)  # Action index 5 is out of bounds for Discrete(5)

def test_terminal_conditions_days():
    # Simulate step until day 0 is reached. Use large inventory to avoid early termination.
    large_env = DynamicPricingEnv(max_inventory=10000, max_days=30, base_arrivals=15)
    obs, info = large_env.reset(seed=42)
    terminated = False
    
    for _ in range(30):
        obs, reward, terminated, truncated, info = large_env.step(2)
        if terminated:
            break
            
    assert terminated is True
    assert large_env.days_until_departure == 0

def test_terminal_conditions_inventory(env):
    # Setup small inventory env to force terminal state via ticket exhaustion
    small_env = DynamicPricingEnv(max_inventory=1, max_days=30, base_arrivals=100)
    obs, info = small_env.reset()
    
    # Pricing action 0 (price 80) which has high purchase probability
    terminated = False
    for _ in range(30):
        obs, reward, terminated, truncated, info = small_env.step(0)
        if terminated:
            break
            
    assert terminated is True
    assert small_env.remaining_inventory == 0
