import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
import gymnasium as gym

# Define MLP class (same as used in training)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Define Actor-Critic model (same as used in training)
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred

# Load the trained model
def load_trained_model(model_path, input_dim, hidden_dim, output_dim):
    # Recreate the model architecture
    actor = MLP(input_dim, hidden_dim, output_dim)
    critic = MLP(input_dim, hidden_dim, 1)
    policy = ActorCritic(actor, critic)

    # Load the saved state_dict
    policy.load_state_dict(torch.load(model_path))
    policy.eval()  # Set to evaluation mode for inference
    return policy

# Simulate an episode and record it as a video
def simulate_and_record(env_name, policy, video_folder):
    # Create a directory to save videos
    os.makedirs(video_folder, exist_ok=True)

    # Wrap the environment for video recording
    env = RecordVideo(
        gym.make(env_name, render_mode="rgb_array"),
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,  # Record every episode
        disable_logger=True,
    )

    # Simulate one episode
    state, _ = env.reset()
    done = False

    while not done:
        # Convert state to tensor for the policy network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_pred, _ = policy(state_tensor)
            action_prob = F.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1).item()

        # Step through the environment
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

    env.close()
    print(f"Video saved in {video_folder}")

# Main function to load model and record video
def main():
    # Model parameters (must match those used during training)
    INPUT_DIM = 8  # LunarLander observation space size
    HIDDEN_DIM = 128
    OUTPUT_DIM = 4  # LunarLander action space size

    # Path to the saved model
    model_path = "lunar_lander_policy.pth"

    # Load the trained model
    policy = load_trained_model(model_path, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

    # Simulate and record video
    env_name = "LunarLander-v3"
    video_folder = "./videos"  # Directory where videos will be saved

    simulate_and_record(env_name, policy, video_folder)

# Run the main function
if __name__ == "__main__":
    main()
