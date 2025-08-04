import pandas as pd
import matplotlib.pyplot as plt
import argparse

def analyze_logs(log_files):
    plt.figure(figsize=(15, 5))

    for i, log_file in enumerate(log_files):
        df = pd.read_csv(log_file)

        # Calculate rolling average for reward
        df['Reward_MA'] = df['Total Reward'].rolling(window=50, min_periods=1).mean()

        # Calculate win rate
        df['Win'] = (df['Result'] == 'victory').astype(int)
        df['Win_Rate_MA'] = df['Win'].rolling(window=50, min_periods=1).mean() * 100

        # Plotting
        # Subplot 1: Smoothed Reward
        plt.subplot(1, 2, 1)
        plt.plot(df['Episode'], df['Reward_MA'], label=f'{log_file} (Smoothed)')
        plt.title('Average Reward per Episode (Smoothed)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.legend()

        # Subplot 2: Win Rate
        plt.subplot(1, 2, 2)
        plt.plot(df['Episode'], df['Win_Rate_MA'], label=f'{log_file} (Win Rate)')
        plt.title('Win Rate (% over last 50 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot training logs.')
    parser.add_argument('log_files', nargs='+', help='List of log files to compare.')
    args = parser.parse_args()
    analyze_logs(args.log_files)
