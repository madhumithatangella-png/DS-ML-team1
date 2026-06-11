import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.simulation.demand_engine import DemandSimulationEngine

class DynamicPricingEnv(gym.Env):
    """
    Custom Gymnasium Environment for AI-Powered Dynamic Pricing in Travel & Hospitality.
    
    State Space:
        [remaining_inventory, days_until_departure, competitor_price, seasonality_factor, market_demand_level]
        - remaining_inventory (float): count of remaining tickets [0, max_inventory]
        - days_until_departure (float): days left to sell tickets [0, max_days]
        - competitor_price (float): competitor price on the day
        - seasonality_factor (float): multiplier indicating weekend/holiday/peak [0.5, 3.0]
        - market_demand_level (float): level of overall market demand {0: Low, 1: Medium, 2: High, 3: Peak}

    Action Space:
        Discrete(5) representing five pricing levels: [80, 100, 120, 140, 160]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_inventory=100, max_days=30, base_arrivals=15):
        super().__init__()
        
        self.max_inventory = max_inventory
        self.max_days = max_days
        self.base_arrivals = base_arrivals
        
        # Initialize demand engine
        self.demand_engine = DemandSimulationEngine(max_days=max_days, base_arrivals=base_arrivals)
        
        # Action space: Discrete(5) -> maps to [80, 100, 120, 140, 160]
        self.pricing_levels = [80, 100, 120, 140, 160]
        self.action_space = spaces.Discrete(len(self.pricing_levels))
        
        # State bounds
        # [remaining_inventory, days_until_departure, competitor_price, seasonality_factor, market_demand_level]
        low = np.array([0.0, 0.0, 50.0, 0.5, 0.0], dtype=np.float32)
        high = np.array([float(max_inventory), float(max_days), 250.0, 3.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Market transition matrix (Markov process for market demand level)
        self.market_transition_matrix = np.array([
            [0.70, 0.25, 0.05, 0.00],  # From Low (0)
            [0.15, 0.70, 0.12, 0.03],  # From Medium (1)
            [0.02, 0.18, 0.70, 0.10],  # From High (2)
            [0.00, 0.05, 0.25, 0.70]   # From Peak (3)
        ])
        
        # State variables
        self.remaining_inventory = None
        self.days_until_departure = None
        self.competitor_price = None
        self.seasonality_factor = None
        self.market_demand_level = None
        
        # Trackers
        self.cumulative_revenue = 0.0
        self.seasonality_schedule = {}

    def _get_obs(self):
        return np.array([
            float(self.remaining_inventory),
            float(self.days_until_departure),
            float(self.competitor_price),
            float(self.seasonality_factor),
            float(self.market_demand_level)
        ], dtype=np.float32)

    def _generate_seasonality_schedule(self):
        """
        Generate seasonality factor schedule for the entire booking window.
        Deterministic weekends, stochastic holiday occurrences, and peak end-period.
        """
        schedule = {}
        for d in range(self.max_days + 1):
            day_idx = self.max_days - d
            # Periodicity: 5th and 6th days of week are Weekend
            if day_idx % 7 in [5, 6]:
                schedule[d] = (1.2, "Weekend")
            # Urgency Peak: Last 15% of the booking period (e.g. last 4 days out of 30)
            elif d <= max(3, int(self.max_days * 0.15)):
                schedule[d] = (1.8, "Peak")
            # Stochastic Holiday: 5% chance on other days
            elif self.np_random.random() < 0.05:
                schedule[d] = (1.5, "Holiday")
            else:
                schedule[d] = (1.0, "Normal")
        return schedule

    def reset(self, seed=None, options=None):
        """
        Reset the environment state for a new episode.
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.remaining_inventory = self.max_inventory
        self.days_until_departure = self.max_days
        
        # Start competitor price stochastically between 90 and 140 (discrete options)
        competitor_options = [90, 100, 110, 120, 130, 140]
        self.competitor_price = float(self.np_random.choice(competitor_options))
        
        # Pre-generate seasonality schedule
        self.seasonality_schedule = self._generate_seasonality_schedule()
        self.seasonality_factor, _ = self.seasonality_schedule[self.days_until_departure]
        
        # Initial market demand level: weighted random initialization
        self.market_demand_level = self.np_random.choice([0, 1, 2, 3], p=[0.15, 0.50, 0.25, 0.10])
        
        # Reset trackers
        self.cumulative_revenue = 0.0
        
        obs = self._get_obs()
        info = {
            "cumulative_revenue": self.cumulative_revenue,
            "seasonality_label": self.seasonality_schedule[self.days_until_departure][1]
        }
        return obs, info

    def step(self, action):
        """
        Execute one action in the environment.
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in space {self.action_space}")
            
        selected_price = self.pricing_levels[action]
        
        # Retrieve current seasonality
        seasonality_factor, seasonality_label = self.seasonality_schedule[self.days_until_departure]
        
        # Simulate daily demand
        tickets_sold, segment_purchases, arrivals = self.demand_engine.simulate_daily_demand(
            selected_price=selected_price,
            competitor_price=self.competitor_price,
            days_left=self.days_until_departure,
            seasonality_factor=seasonality_factor,
            market_demand_level=self.market_demand_level
        )
        
        # Cap sales by remaining inventory
        if tickets_sold > self.remaining_inventory:
            actual_sold = self.remaining_inventory
            # Re-scale segment purchases to match capped inventory
            if tickets_sold > 0:
                scale = actual_sold / tickets_sold
                for seg in segment_purchases:
                    segment_purchases[seg] = int(round(segment_purchases[seg] * scale))
                # Adjust rounding errors
                diff = actual_sold - sum(segment_purchases.values())
                if diff != 0:
                    for seg in segment_purchases:
                        if segment_purchases[seg] + diff >= 0:
                            segment_purchases[seg] += diff
                            break
            tickets_sold = actual_sold
        else:
            actual_sold = tickets_sold
            
        # Update state variables
        self.remaining_inventory -= actual_sold
        self.days_until_departure -= 1
        
        # Calculate rewards
        daily_revenue = float(actual_sold * selected_price)
        self.cumulative_revenue += daily_revenue
        reward = daily_revenue
        
        # Check terminations
        terminated = (self.remaining_inventory <= 0) or (self.days_until_departure <= 0)
        truncated = False
        
        # Transitions for the next step (if not terminated)
        if not terminated:
            # 1. Update competitor price stochastically
            self.competitor_price += float(self.np_random.choice([-10.0, 0.0, 10.0], p=[0.25, 0.50, 0.25]))
            self.competitor_price = float(np.clip(self.competitor_price, 90.0, 140.0))
            
            # 2. Update seasonality factor
            self.seasonality_factor, seasonality_label = self.seasonality_schedule[self.days_until_departure]
            
            # 3. Transition market demand level
            self.market_demand_level = self.np_random.choice(
                [0, 1, 2, 3], 
                p=self.market_transition_matrix[self.market_demand_level]
            )
        else:
            # Set to final values at termination for consistency
            self.days_until_departure = max(0, self.days_until_departure)
            self.seasonality_factor = 1.0
            seasonality_label = "Terminal"
            
        obs = self._get_obs()
        
        market_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Peak"}
        info = {
            "tickets_sold": actual_sold,
            "selected_price": selected_price,
            "competitor_price": self.competitor_price,
            "daily_revenue": daily_revenue,
            "cumulative_revenue": self.cumulative_revenue,
            "segment_purchases": segment_purchases,
            "arrivals_breakdown": arrivals,
            "seasonality_label": seasonality_label,
            "market_demand_label": market_labels.get(self.market_demand_level, "Unknown")
        }
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render environment state.
        """
        market_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Peak"}
        print(
            f"Day: {self.max_days - self.days_until_departure}/{self.max_days} | "
            f"Inv: {self.remaining_inventory} | "
            f"Comp: ${self.competitor_price:.2f} | "
            f"Seasonality Factor: {self.seasonality_factor:.1f} | "
            f"Market Demand: {market_labels.get(self.market_demand_level, 'Unknown')} | "
            f"Cum. Revenue: ${self.cumulative_revenue:.2f}"
        )
