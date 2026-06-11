# AI-Powered Dynamic Pricing System using Reinforcement Learning for Travel & Hospitality

An enterprise-ready, modular dynamic pricing simulator built using Gymnasium and NumPy. This project designs, implements, and simulates the Markov Decision Process (MDP) for optimizing ticket pricing under perishable inventory, competitor fluctuations, and stochastic customer demand.

---

## 1. Project Overview

In travel and hospitality (e.g., airlines, hotels, trains), inventory is **perishable**—once the departure date or booking window closes, unsold seats or rooms hold zero value. This project models the pricing optimization task as a Reinforcement Learning (RL) environment. By adjusting pricing dynamically in response to remaining capacity, time left, competitor pricing, seasonality, and market demand, the system aims to maximize total ticket revenue.

---

## 2. Business Problem

Dynamic pricing is the strategy of adjusting prices in real-time to match market conditions, competitor movements, and customer willingness to pay.
* **Perishability**: Inventory expires at departure.
* **Price Sensitivity**: Lowering prices stimulates demand, while raising prices increases margins but risks unsold inventory.
* **Competition**: Customers compare prices; if a competitor is significantly cheaper, price-sensitive segments will book elsewhere.
* **Urgency**: As departure approaches, certain customers (e.g., business travelers) become less price-sensitive and buy tickets at premium rates.
* **Seasonality**: Demand surges during weekends, holidays, and peak vacation periods.

---

## 3. Markov Decision Process (MDP) Formulation

The dynamic pricing problem is formulated as a finite-horizon MDP:

### State Space ($S$)
The state is represented as a 5-dimensional vector:
$$S = [I_t, D_t, C_t, F_t, M_t]$$

where:
1. **$I_t$ (Remaining Inventory)**: Float in range `[0, max_inventory]`. Represents the remaining seat/room capacity.
2. **$D_t$ (Days Until Departure)**: Float in range `[0, max_days]`. Represents the booking window countdown.
3. **$C_t$ (Competitor Price)**: Float in range `[90.0, 140.0]`. Fluctuates stochastically.
4. **$F_t$ (Seasonality Factor)**: Float multiplier indicating weekend, holiday, peak, or normal travel season.
5. **$M_t$ (Market Demand Level)**: Integer representing overall market demand:
   * `0` = Low (Multiplier: `0.6`)
   * `1` = Medium (Multiplier: `1.0`)
   * `2` = High (Multiplier: `1.4`)
   * `3` = Peak (Multiplier: `1.8`)

### Action Space ($A$)
A Gymnasium `Discrete(5)` action space representing discrete price levels:
$$A \in \{80, 100, 120, 140, 160\}$$

### Reward Function ($R$)
The reward at time step $t$ is the revenue earned from sales on that day:
$$R_t = \text{tickets\_sold}_t \times \text{selected\_price}$$
Both daily revenue and cumulative revenue are tracked.

### Terminal Conditions
An episode terminates (ends) when:
1. **$I_t = 0$** (Inventory is completely exhausted).
2. **$D_t = 0$** (Departure date reached).

---

## 4. Customer Segmentation

Customer arrivals are simulated stochastically, split into three distinct segments:

| Customer Segment | Proportion | Price Sensitivity | Competitor Influence | Urgency Sensitivity |
| :--- | :---: | :---: | :---: | :---: |
| **Budget** | 50% | **High** | **Strong** | **Low** |
| **Regular** | 35% | **Medium** | **Medium** | **Medium** |
| **Premium** | 15% | **Low** | **Minimal** | **High** |

### Purchase Probability Model
For each customer, the decision to purchase is determined by a stochastic probability formula:
$$P(\text{Purchase}) = \text{Base Demand} \times f(\text{Price}) \times f(\text{Competitor}) \times f(\text{Urgency}) \times F_t \times M_t$$

Where:
* $f(\text{Price}) = \exp(-\beta_{\text{price}} \cdot \text{Normalized Price})$
* $f(\text{Competitor}) = \exp\left(\beta_{\text{comp}} \cdot \frac{\text{Competitor Price} - \text{Selected Price}}{100}\right)$
* $f(\text{Urgency}) = \exp\left(\beta_{\text{urgency}} \cdot \frac{\text{max\_days} - D_t}{\text{max\_days}}\right)$

---

## 5. Competitor Pricing, Seasonality & Market Demand

### Competitor pricing
Competitor prices undergo a daily random walk centered on baseline ranges:
$$C_{t+1} = \text{clip}(C_t + \epsilon_t, 90.0, 140.0)$$
where $\epsilon_t \in \{-10.0, 0.0, 10.0\}$ with probabilities $[0.25, 0.50, 0.25]$ respectively.

### Seasonality Engine
Seasonality is modeled deterministically and stochastically across the booking window:
* **Normal**: Factor `1.0`
* **Weekend** (every 7 days): Factor `1.2`
* **Holiday** (stochastic 5% chance): Factor `1.5`
* **Peak Season** (last 15% booking window): Factor `1.8`

### Market Demand Engine
Market demand level $M_t$ is simulated as a Markov chain with self-reinforcing transitions, ensuring smooth economic waves:
* Probability of staying in the same level is high ($70\%$).
* Shifts to adjacent demand levels happen smoothly.

---

## 6. Project Directory Structure

```
dynamic-pricing-rl/
│
├── data/                       # Cached simulation data and configurations
│   └── .gitkeep
├── reports/                    # Generated charts and reports
│   └── figures/
│       └── .gitkeep
├── notebooks/                  # Jupyter notebooks for interactive analysis
│   └── .gitkeep
├── src/                        # Source Code
│   ├── __init__.py
│   ├── environment/            # Gymnasium Environment
│   │   ├── __init__.py
│   │   └── dynamic_pricing_env.py
│   ├── simulation/             # Stochastic Demand Engine
│   │   ├── __init__.py
│   │   └── demand_engine.py
│   ├── visualization/          # Matplotlib Plotting script
│   │   ├── __init__.py
│   │   └── plots.py
│   └── utils/                  # Helper utilities
│       └── __init__.py
│
├── tests/                      # Automated test suite (pytest)
│   ├── __init__.py
│   ├── test_demand_engine.py
│   └── test_environment.py
│
├── README.md                   # Documentation
├── requirements.txt            # Python Dependencies
└── .gitignore                  # Gitignore file
```

---

## 7. Installation & Getting Started

### 1. Set Up Environment
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 8. Running Simulations and Visualizations

To simulate the system under a baseline random policy and generate all diagnostic dashboard plots:

```bash
python src/visualization/plots.py
```

This generates and saves **7 charts** inside `reports/figures/`:
1. **`demand_vs_price.png`**: Expected customer segment demand across pricing options.
2. **`revenue_vs_price.png`**: Revenue yields under different pricing strategies.
3. **`inventory_reduction_over_time.png`**: Trajectories of ticket sales.
4. **`competitor_price_impact.png`**: Market capture reactions to competitor price changes.
5. **`seasonality_impact.png`**: Sales response to weekend/holiday demand spikes.
6. **`customer_segment_distribution.png`**: Breakdown of overall customer types.
7. **`market_demand_distribution.png`**: Distribution of overall market states.

---

## 9. Running Automated Tests

To verify environment logic and demand engine calculations, run the pytest suite:

```bash
pytest tests/
```

This verifies:
* Environment resets and observations.
* Action out-of-bounds boundary errors.
* Inventory depletion and episode terminations.
* Custom customer segment sensitivities.
* Competitor price reactions.
* Seasonality multiplier effects.

---

## 10. Future Work (Phase 2)

In Phase 2, we will integrate reinforcement learning agents to solve the MDP:
* **Tabular Q-Learning**: For baseline discrete state approximations.
* **Deep Q-Networks (DQN)**: For continuous state-space modeling.
* **Proximal Policy Optimization (PPO)**: To test continuous actions and policy gradient solutions.
* **Streamlit Dashboard**: A live web dashboard to visualize agent pricing choices in real-time.
