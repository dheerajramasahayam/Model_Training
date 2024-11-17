import tensorflow as tf
from src.model import create_model
from src.agent import DQNAgent
from src.utils import calculate_reward, one_hot_encode_password
from tqdm import tqdm
import numpy as np

def train_models(processed_handshakes, passwords, char_space):
    # --- Hyperparameters ---
    batch_size = 32
    epochs = 100  # Adjust as needed
    learning_rate = 0.001
    gamma = 0.95  # Discount factor for DQN
    epsilon = 1.0  # Exploration rate for DQN
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # --- Model Creation ---

    # Create the Transformer model
    input_shape = processed_handshakes[0].shape
    transformer_model = create_model(input_shape, char_space)

    # Create the DQN agent
    state_size = 128  # Match the embedding dimension in your Transformer
    dqn_agent = DQNAgent(state_size=state_size, action_size=len(char_space))

    # --- Training Loop ---

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    print("Training the model...")
    for epoch in range(epochs):
        with tqdm(total=len(processed_handshakes), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in range(len(processed_handshakes) // batch_size):
                # Get a batch of data
                batch_handshakes = processed_handshakes[batch * batch_size : (batch + 1) * batch_size]
                batch_passwords = passwords[batch * batch_size : (batch + 1) * batch_size]

                # --- Transformer Prediction ---
                with tf.GradientTape() as tape:
                    # Get predictions from the Transformer
                    predictions = transformer_model(np.array(batch_handshakes))

                    # --- DQN Action Selection ---
                    dqn_actions = []
                    for i in range(batch_size):
                        # Get DQN action (character index)
                        action = dqn_agent.act(predictions[i])
                        dqn_actions.append(action)

                    # --- Reward Calculation ---
                    rewards = []
                    predicted_passwords = []
                    for i in range(batch_size):
                        # Convert action to character
                        predicted_char = char_space[dqn_actions[i]]
                        predicted_passwords.append(predicted_char)

                        # Calculate reward (replace with your actual reward function)
                        reward = calculate_reward(predicted_char, batch_passwords[i])
                        rewards.append(reward)

                    # --- DQN Experience Replay ---
                    for i in range(batch_size):
                        # Store experience in the DQN agent's memory
                        # (Assuming next_state is the same as current state for simplicity)
                        dqn_agent.remember(predictions[i], dqn_actions[i], rewards[i], predictions[i], False)

                    # Update the DQN agent (replay experiences)
                    dqn_agent.replay(batch_size)

                    # --- Transformer Loss and Optimization ---
                    # One-hot encode the actual passwords
                    one_hot_passwords = np.array([one_hot_encode_password(pwd, char_space) for pwd in batch_passwords])

                    # Calculate loss
                    loss = loss_fn(one_hot_passwords, predictions)

                # Apply gradients
                gradients = tape.gradient(loss, transformer_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))

                # --- Update progress bar and display results ---
                pbar.update(batch_size)
                pbar.set_postfix({"Loss": f"{loss.numpy():.4f}", "Avg Reward": f"{np.mean(rewards):.2f}"})

                # Print some predicted passwords (for monitoring)
                if batch % 10 == 0:  # Print every 10 batches
                    print(f"Predicted Passwords: {predicted_passwords[:5]}")  # Print first 5

    # --- Save the trained models ---
    transformer_model.save('model/transformer')
    dqn_agent.model.save('model/dqn')