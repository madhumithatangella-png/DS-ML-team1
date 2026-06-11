import os
import matplotlib.pyplot as plt
import numpy as np
from src.environment.dynamic_pricing_env import DynamicPricingEnv
from src.simulation.demand_engine import DemandSimulationEngine

# Set style for premium visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def generate_demand_vs_price(engine, save_path):
    """Plot 1: Demand (Purchase Probability) vs Selected Price for each segment."""
    prices = np.linspace(80, 160, 100)
    segments = ["Budget", "Regular", "Premium"]
    colors = {"Budget": "#e74c3c", "Regular": "#3498db", "Premium": "#2ecc71"}
    
    plt.figure(figsize=(8, 5))
    for seg in segments:
        probs = [
            engine.get_purchase_probability(
                segment=seg,
                selected_price=p,
                competitor_price=120,
                days_left=15,
                seasonality_factor=1.0,
                market_factor=1.0
            ) for p in prices
        ]
        plt.plot(prices, probs, label=f"{seg} Customer", color=colors[seg], linewidth=2.5)
        
    plt.title("Purchase Probability vs. Selected Price\n(Competitor Price: $120, Days Left: 15, Normal Season)", pad=15)
    plt.xlabel("Selected Price ($)")
    plt.ylabel("Purchase Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "demand_vs_price.png"), dpi=150)
    plt.close()

def generate_revenue_vs_price(engine, save_path):
    """Plot 2: Expected Revenue vs Price under baseline conditions."""
    prices = [80, 100, 120, 140, 160]
    expected_revenues = []
    
    # Run 500 simulations per price point to get smooth average revenue
    for p in prices:
        revs = []
        for _ in range(500):
            # Assume 15 days left, competitor at 120, normal season, medium market demand
            sold, _, _ = engine.simulate_daily_demand(
                selected_price=p,
                competitor_price=120,
                days_left=15,
                seasonality_factor=1.0,
                market_demand_level=1
            )
            revs.append(sold * p)
        expected_revenues.append(np.mean(revs))
        
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        [str(p) for p in prices], 
        expected_revenues, 
        color="#9b59b6", 
        edgecolor='none', 
        width=0.5,
        alpha=0.85
    )
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 2, 
            f"${height:.1f}", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
        
    plt.title("Expected Daily Revenue vs. Pricing Levels\n(Competitor Price: $120, Days Left: 15, Normal Season)", pad=15)
    plt.xlabel("Pricing Action Level ($)")
    plt.ylabel("Expected Daily Revenue ($)")
    plt.ylim(0, max(expected_revenues) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "revenue_vs_price.png"), dpi=150)
    plt.close()

def generate_inventory_reduction(env, save_path):
    """Plot 3: Inventory depletion paths over time under random pricing policy."""
    plt.figure(figsize=(9, 5))
    
    # Run 5 separate random-policy simulations
    for run in range(1, 6):
        obs, info = env.reset(seed=42 + run)
        inventory_path = [env.remaining_inventory]
        days = [env.max_days - env.days_until_departure]
        
        terminated = False
        while not terminated:
            # Random action selection
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            inventory_path.append(env.remaining_inventory)
            days.append(env.max_days - env.days_until_departure)
            
        plt.plot(days, inventory_path, marker='o', linestyle='-', alpha=0.75, linewidth=2, label=f"Simulation {run}")
        
    plt.title("Inventory Depletion Over Time\n(Random Pricing Policy, 30-Day Window)", pad=15)
    plt.xlabel("Days Elapsed")
    plt.ylabel("Remaining Ticket Inventory")
    plt.xlim(-0.5, env.max_days + 0.5)
    plt.ylim(-5, env.max_inventory + 5)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "inventory_reduction_over_time.png"), dpi=150)
    plt.close()

def generate_competitor_impact(engine, save_path):
    """Plot 4: Purchase Probability vs Competitor Price differential (Comp - Our Price)."""
    comp_prices = np.linspace(80, 160, 100)
    our_price = 120
    segments = ["Budget", "Regular", "Premium"]
    colors = {"Budget": "#e74c3c", "Regular": "#3498db", "Premium": "#2ecc71"}
    
    plt.figure(figsize=(8, 5))
    for seg in segments:
        probs = [
            engine.get_purchase_probability(
                segment=seg,
                selected_price=our_price,
                competitor_price=cp,
                days_left=15,
                seasonality_factor=1.0,
                market_factor=1.0
            ) for cp in comp_prices
        ]
        diffs = comp_prices - our_price
        plt.plot(diffs, probs, label=f"{seg} Customer", color=colors[seg], linewidth=2.5)
        
    plt.axvline(0, color='grey', linestyle='--', alpha=0.7)
    plt.title("Purchase Probability vs. Competitor Price Differential\n(Our Price: $120, Days Left: 15, Normal Season)", pad=15)
    plt.xlabel("Competitor Price Differential (Competitor Price - Our Price) ($)")
    plt.ylabel("Purchase Probability")
    plt.ylim(-0.05, 1.05)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "competitor_price_impact.png"), dpi=150)
    plt.close()

def generate_seasonality_impact(engine, save_path):
    """Plot 5: Expected demand under different seasonality factors."""
    seasons = ["Normal", "Weekend", "Holiday", "Peak"]
    factors = [1.0, 1.2, 1.5, 1.8]
    expected_sales = []
    
    for fact in factors:
        sales = []
        for _ in range(500):
            sold, _, _ = engine.simulate_daily_demand(
                selected_price=120,
                competitor_price=120,
                days_left=15,
                seasonality_factor=fact,
                market_demand_level=1
            )
            sales.append(sold)
        expected_sales.append(np.mean(sales))
        
    plt.figure(figsize=(8, 5))
    bars = plt.bar(seasons, expected_sales, color="#34495e", edgecolor='none', width=0.5, alpha=0.85)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 0.1, 
            f"{height:.2f}", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
        
    plt.title("Expected Daily Ticket Sales by Seasonality Factor\n(Our Price: $120, Competitor Price: $120, Medium Market Demand)", pad=15)
    plt.xlabel("Seasonality Class")
    plt.ylabel("Expected Daily Tickets Sold")
    plt.ylim(0, max(expected_sales) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "seasonality_impact.png"), dpi=150)
    plt.close()

def generate_customer_segment_distribution(engine, save_path):
    """Plot 6: Customer Segment Distribution in overall arrivals."""
    # Run a long series of arrivals to gather distribution statistics
    np.random.seed(42)
    segment_counts = {"Budget": 0, "Regular": 0, "Premium": 0}
    
    for _ in range(1000):
        # Trigger simulation of single step customer segments
        _, _, arrivals = engine.simulate_daily_demand(
            selected_price=120,
            competitor_price=120,
            days_left=15,
            seasonality_factor=1.0,
            market_demand_level=1
        )
        for seg in segment_counts:
            segment_counts[seg] += arrivals[seg]
            
    total_arrivals = sum(segment_counts.values())
    percentages = [segment_counts[seg] / total_arrivals * 100 for seg in ["Budget", "Regular", "Premium"]]
    
    plt.figure(figsize=(6, 5))
    plt.pie(
        percentages, 
        labels=["Budget (50%)", "Regular (35%)", "Premium (15%)"], 
        colors=["#e74c3c", "#3498db", "#2ecc71"],
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'none', 'linewidth': 0, 'antialiased': True}
    )
    plt.title("Observed Customer Segment Arrival Distribution", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "customer_segment_distribution.png"), dpi=150)
    plt.close()

def generate_market_demand_distribution(env, save_path):
    """Plot 7: Market Demand Distribution across many episodes."""
    np.random.seed(42)
    demand_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for _ in range(200):
        env.reset()
        demand_counts[env.market_demand_level] += 1
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if not terminated:
                demand_counts[env.market_demand_level] += 1
                
    total_steps = sum(demand_counts.values())
    levels = ["Low", "Medium", "High", "Peak"]
    counts = [demand_counts[i] for i in range(4)]
    percentages = [c / total_steps * 100 for c in counts]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(levels, percentages, color="#16a085", edgecolor='none', width=0.5, alpha=0.85)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.0, 
            height + 1.0, 
            f"{height:.1f}%", 
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
        
    plt.title("Market Demand Level Distribution\n(Markov Chain Transitions Over 200 Episodes)", pad=15)
    plt.xlabel("Market Demand Level")
    plt.ylabel("Frequency (%)")
    plt.ylim(0, max(percentages) * 1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "market_demand_distribution.png"), dpi=150)
    plt.close()

def run_all_plots():
    save_path = os.path.join("reports", "figures")
    create_directory(save_path)
    print(f"Generating dashboard plots and saving to {save_path}...")
    
    engine = DemandSimulationEngine()
    env = DynamicPricingEnv()
    
    generate_demand_vs_price(engine, save_path)
    print("1/7: Generated demand_vs_price.png")
    
    generate_revenue_vs_price(engine, save_path)
    print("2/7: Generated revenue_vs_price.png")
    
    generate_inventory_reduction(env, save_path)
    print("3/7: Generated inventory_reduction_over_time.png")
    
    generate_competitor_impact(engine, save_path)
    print("4/7: Generated competitor_price_impact.png")
    
    generate_seasonality_impact(engine, save_path)
    print("5/7: Generated seasonality_impact.png")
    
    generate_customer_segment_distribution(engine, save_path)
    print("6/7: Generated customer_segment_distribution.png")
    
    generate_market_demand_distribution(env, save_path)
    print("7/7: Generated market_demand_distribution.png")
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    run_all_plots()
