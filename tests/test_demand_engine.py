import pytest
import numpy as np
from src.simulation.demand_engine import DemandSimulationEngine

@pytest.fixture
def demand_engine():
    return DemandSimulationEngine(max_days=30, base_arrivals=15)

def test_price_sensitivity(demand_engine):
    # Budget should be much more sensitive to price increases than Premium
    # At low price (80), purchase probability should be higher than at high price (160)
    prob_budget_low = demand_engine.get_purchase_probability("Budget", 80, 120, 15, 1.0, 1.0)
    prob_budget_high = demand_engine.get_purchase_probability("Budget", 160, 120, 15, 1.0, 1.0)
    
    assert prob_budget_low > prob_budget_high
    
    prob_premium_low = demand_engine.get_purchase_probability("Premium", 80, 120, 15, 1.0, 1.0)
    prob_premium_high = demand_engine.get_purchase_probability("Premium", 160, 120, 15, 1.0, 1.0)
    
    assert prob_premium_low >= prob_premium_high
    # Budget sensitivity ratio of decay should be greater than Premium's decay
    ratio_budget = prob_budget_low / (prob_budget_high + 1e-9)
    ratio_premium = prob_premium_low / (prob_premium_high + 1e-9)
    
    assert ratio_budget > ratio_premium

def test_competitor_price_impact(demand_engine):
    # If competitor is cheap (90) vs expensive (140) while our price is 120,
    # purchase probability should be higher when competitor is expensive.
    prob_comp_cheap = demand_engine.get_purchase_probability("Budget", 120, 90, 15, 1.0, 1.0)
    prob_comp_expensive = demand_engine.get_purchase_probability("Budget", 120, 140, 15, 1.0, 1.0)
    
    assert prob_comp_expensive > prob_comp_cheap
    
    # Premium should not care much about competitor pricing
    prob_prem_cheap = demand_engine.get_purchase_probability("Premium", 120, 90, 15, 1.0, 1.0)
    prob_prem_expensive = demand_engine.get_purchase_probability("Premium", 120, 140, 15, 1.0, 1.0)
    
    diff_prem = abs(prob_prem_expensive - prob_prem_cheap)
    diff_budget = abs(prob_comp_expensive - prob_comp_cheap)
    assert diff_budget > diff_prem

def test_urgency_impact(demand_engine):
    # As days_left decreases (urgency increases), purchase probability should increase or stay same.
    # Premium segment is highly sensitive to urgency.
    prob_far = demand_engine.get_purchase_probability("Premium", 120, 120, 30, 1.0, 1.0)
    prob_near = demand_engine.get_purchase_probability("Premium", 120, 120, 1, 1.0, 1.0) # 1 day left
    
    assert prob_near > prob_far
    
    # Budget segment has low urgency sensitivity
    prob_far_budget = demand_engine.get_purchase_probability("Budget", 120, 120, 30, 1.0, 1.0)
    prob_near_budget = demand_engine.get_purchase_probability("Budget", 120, 120, 1, 1.0, 1.0)
    
    ratio_prem = prob_near / (prob_far + 1e-9)
    ratio_budget = prob_near_budget / (prob_far_budget + 1e-9)
    assert ratio_prem > ratio_budget

def test_simulate_daily_demand(demand_engine):
    np.random.seed(42)
    tickets_sold, segment_purchases, arrivals = demand_engine.simulate_daily_demand(
        selected_price=120,
        competitor_price=120,
        days_left=15,
        seasonality_factor=1.0,
        market_demand_level=1
    )
    
    # Assert type and structure of returns
    assert isinstance(tickets_sold, int)
    assert isinstance(segment_purchases, dict)
    assert isinstance(arrivals, dict)
    
    for seg in ["Budget", "Regular", "Premium"]:
        assert seg in segment_purchases
        assert seg in arrivals
        assert segment_purchases[seg] <= arrivals[seg]
        
    assert sum(segment_purchases.values()) == tickets_sold
