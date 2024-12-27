import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Simulation Parameters
N = 10  # Number of agents
T = 100  # Number of time steps
kappa = 1.0  # Strength of interaction
p = 2  # Decay exponent
eta_scale = 0.1  # Noise scale

# Initialize agent states
np.random.seed(42)
agents = {
    i: {
        'wealth': np.random.uniform(10, 100),
        'holdings': np.random.uniform(1, 10),
        'information': np.random.uniform(0, 1)
    }
    for i in range(N)
}

# State vector for visualization
def agent_state_vector(agent):
    return np.array([agent['wealth'], agent['holdings'], agent['information']])

# Compute pairwise potential
def compute_potential(agent_i, agent_j):
    state_i = agent_state_vector(agent_i)
    state_j = agent_state_vector(agent_j)
    distance = np.linalg.norm(state_i - state_j)
    return kappa / (distance ** p + 1e-9)  # Avoid division by zero

# Update dynamics (Langevin)
def update_states(agents, eta_scale):
    updated_agents = {}
    for i, agent_i in agents.items():
        force = np.zeros(3)  # Gradient of potential
        for j, agent_j in agents.items():
            if i != j:
                state_i = agent_state_vector(agent_i)
                state_j = agent_state_vector(agent_j)
                distance_vector = state_i - state_j
                distance = np.linalg.norm(distance_vector)
                force -= kappa * distance_vector / (distance ** (p + 2) + 1e-9)
        
        # Add stochastic noise
        noise = eta_scale * np.random.normal(size=3)
        updated_state = agent_state_vector(agent_i) + force + noise
        updated_agents[i] = {
            'wealth': updated_state[0],
            'holdings': updated_state[1],
            'information': updated_state[2],
        }
    return updated_agents

# Simulation and Visualization
agent_positions = {i: (np.random.uniform(0, 10), np.random.uniform(0, 10)) for i in range(N)}
plt.figure(figsize=(8, 6))
for t in range(T):
    if t % 20 == 0 or t == T - 1:
        # Build interaction network
        G = nx.Graph()
        for i in agents:
            G.add_node(i, pos=agent_positions[i])
        for i in agents:
            for j in agents:
                if i != j:
                    weight = compute_potential(agents[i], agents[j])
                    if weight > 0.1:  # Only show significant interactions
                        G.add_edge(i, j, weight=weight)
        
# Visualization
    # Update states
    agents = update_states(agents, eta_scale)

pos = nx.get_node_attributes(G, 'pos')
weights = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=300, font_size=10)
nx.draw_networkx_edges(
G, pos, edge_color=list(weights.values()), edge_cmap=plt.cm.Blues, width=2
)
plt.title(f"Market Interaction Network at Time {t}")
plt.show()
    
    
