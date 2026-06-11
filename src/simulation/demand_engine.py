import numpy as np

class DemandSimulationEngine:
    """
    Stochastic Demand Simulation Engine for Dynamic Pricing.
    Simulates customer arrivals and purchase decisions based on:
    - Selected Price vs Competitor Price
    - Days until departure (Urgency)
    - Seasonality factors
    - Market demand levels
    - Customer segmentation (Budget, Regular, Premium)
    """
    
    def __init__(self, max_days=30, base_arrivals=15):
        self.max_days = max_days
        self.base_arrivals = base_arrivals
        
        # Segment proportions
        self.segment_ratios = {
            "Budget": 0.50,
            "Regular": 0.35,
            "Premium": 0.15
        }
        
        # Base demand (probability scale) for each segment
        self.base_demand = {
            "Budget": 0.60,
            "Regular": 0.50,
            "Premium": 0.40
        }
        
        # Price sensitivity parameters (exponential decay rate)
        # Higher means more sensitive (probability drops faster with price)
        self.price_sensitivity = {
            "Budget": 2.5,    # Very price sensitive
            "Regular": 1.2,   # Moderate price sensitivity
            "Premium": 0.4    # Low price sensitivity
        }
        
        # Competitor sensitivity: response to competitor price differential
        # competitor_factor = exp(sensitivity * (competitor_price - price) / 100)
        self.competitor_sensitivity = {
            "Budget": 1.8,    # Strongly affected by competitor price
            "Regular": 0.8,   # Moderately affected
            "Premium": 0.1    # Hardly affected
        }
        
        # Urgency sensitivity: response to approaching departure date
        # urgency_factor = exp(sensitivity * (max_days - days_left) / max_days)
        self.urgency_sensitivity = {
            "Budget": 0.2,    # Low urgency impact
            "Regular": 0.6,   # Moderate urgency impact
            "Premium": 1.5    # High urgency sensitivity
        }
        
        # Market demand level multipliers
        self.market_multipliers = {
            0: 0.6,  # Low
            1: 1.0,  # Medium
            2: 1.4,  # High
            3: 1.8   # Peak
        }

    def get_purchase_probability(self, segment, selected_price, competitor_price, days_left, seasonality_factor, market_factor):
        """
        Calculate the stochastic purchase probability for a specific customer segment.
        Formula:
        purchase_probability = base_demand * price_factor * urgency_factor * competitor_factor * seasonality_factor * market_factor
        """
        # 1. Price Factor: Normalize price between min (80) and max (160) price levels
        min_p, max_p = 80, 160
        norm_price = (selected_price - min_p) / (max_p - min_p)
        price_factor = np.exp(-self.price_sensitivity[segment] * norm_price)
        
        # 2. Urgency Factor: Increases as days_left approaches 0
        urgency_frac = (self.max_days - days_left) / self.max_days
        urgency_factor = np.exp(self.urgency_sensitivity[segment] * urgency_frac)
        
        # 3. Competitor Factor: Favorable if competitor_price > selected_price
        # Normalize differential by dividing by 100 to scale nicely
        price_diff = (competitor_price - selected_price) / 100.0
        competitor_factor = np.exp(self.competitor_sensitivity[segment] * price_diff)
        
        # 4. Base probability product
        base_prob = self.base_demand[segment]
        
        # 5. Combined probability
        purchase_prob = (
            base_prob * 
            price_factor * 
            urgency_factor * 
            competitor_factor * 
            seasonality_factor * 
            market_factor
        )
        
        # Ensure probability is strictly bounded between 0.0 and 1.0
        return np.clip(purchase_prob, 0.0, 1.0)

    def simulate_daily_demand(self, selected_price, competitor_price, days_left, seasonality_factor, market_demand_level):
        """
        Simulate customer arrivals and total purchases (tickets sold) for a single day.
        Returns:
            tickets_sold (int): Number of purchases made.
            segment_purchases (dict): Breakdown of purchases by customer segment.
            arrivals_breakdown (dict): Breakdown of total arrivals by customer segment.
        """
        market_factor = self.market_multipliers.get(market_demand_level, 1.0)
        
        # Calculate expected arrivals scaling with seasonality and market demand level
        expected_arrivals = self.base_arrivals * seasonality_factor * market_factor
        
        # Sample arrivals using Poisson distribution
        total_arrivals = np.random.poisson(expected_arrivals)
        
        tickets_sold = 0
        segment_purchases = {"Budget": 0, "Regular": 0, "Premium": 0}
        arrivals_breakdown = {"Budget": 0, "Regular": 0, "Premium": 0}
        
        # Assign segments and determine purchase decisions for each customer
        segments = list(self.segment_ratios.keys())
        probabilities = list(self.segment_ratios.values())
        
        if total_arrivals > 0:
            customer_segments = np.random.choice(segments, size=total_arrivals, p=probabilities)
            
            for seg in customer_segments:
                arrivals_breakdown[seg] += 1
                prob = self.get_purchase_probability(
                    segment=seg,
                    selected_price=selected_price,
                    competitor_price=competitor_price,
                    days_left=days_left,
                    seasonality_factor=seasonality_factor,
                    market_factor=market_factor
                )
                if np.random.rand() < prob:
                    tickets_sold += 1
                    segment_purchases[seg] += 1
                    
        return tickets_sold, segment_purchases, arrivals_breakdown
