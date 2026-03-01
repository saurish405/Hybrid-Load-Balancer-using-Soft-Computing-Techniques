import matplotlib.pyplot as plt
import numpy as np

def plot_server_performance(server_data):
    """
    Expects a list of dictionaries or objects with server ID and tasks processed.
    """
    ids = [f"Server {s.id}\n(Speed {s.speed})" for s in server_data]
    tasks = [s.tasks_processed for s in server_data]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(ids, tasks, color=['#4CAF50', '#2196F3', '#FF9800'])
    
    plt.xlabel('Server Configuration')
    plt.ylabel('Total Tasks Processed')
    plt.title('Intelligent Load Balancer: Workload Distribution')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('plots/performance_results.png')
    print("\n[Success] Plot saved to plots/performance_results.png")
    plt.show()