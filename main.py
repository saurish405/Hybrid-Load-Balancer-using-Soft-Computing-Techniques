import simpy
import pandas as pd
import numpy as np
import os
from src.sim_engine import Server
from src.ann_predictor import LoadPredictor
from src.fuzzy_logic import FuzzyController
from src.utils import plot_server_performance

def main():
    # 1. Initialize
    env = simpy.Environment()
    predictor = LoadPredictor()
    
    # Train if model doesn't exist
    if not os.path.exists('models/predictor.h5'):
        predictor.train_from_csv('data/stress_traffic.csv')
    
    fuzzy = FuzzyController()
    
    # Create 3 Servers with different processing speeds
    servers = [Server(env, 1, 0.5), Server(env, 2, 0.8), Server(env, 3, 1.2)]
    
    traffic_df = pd.read_csv('data/stress_traffic.csv')

    def run_sim(env):
        for i in range(120): # Run for 120 minutes
            requests = traffic_df.iloc[i]['requests']
            
            # Get ANN Prediction (History of last 60 mins)
            history = traffic_df.iloc[max(0, i-60):i]['requests'].tolist()
            if len(history) < 60: history = [50]*60 
            
            trend = predictor.predict_next(history)
            trend_score = min(100, (trend / 150) * 100) # Normalize to 0-100

            # Distribute each request
            for _ in range(requests):
                scores = []
                for s in servers:
                    # Current load = (active tasks / capacity) * 100
                    load = (s.resource.count / s.resource.capacity) * 100
                    scores.append(fuzzy.compute_priority(load, trend_score))
                
                # Pick best server
                best_s = servers[np.argmax(scores)]
                env.process(best_s.handle_task(f"T_{i}"))
            
            yield env.timeout(1)
            if i % 10 == 0: print(f"Minute {i} processed...")

    print("Starting Intelligent Load Balancer Simulation...")
    env.process(run_sim(env))
    env.run()

    print("\n--- Final Stats ---")
    for s in servers:
        print(f"Server {s.id} (Speed {s.speed}): {s.tasks_processed} tasks handled.")

    plot_server_performance(servers)

if __name__ == "__main__":
    main()