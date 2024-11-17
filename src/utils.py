import numpy as np

def calculate_reward(predicted_char, actual_password):
    """
    Calculates the reward for a predicted character.

    Args:
        predicted_char: The character predicted by the AI model.
        actual_password: The actual password.

    Returns:
        The reward value (a float).
    """
    if predicted_char in actual_password:
        # Higher reward if the character is in the correct position
        pos = actual_password.index(predicted_char)
        reward = 1 + (len(actual_password) - pos) / len(actual_password)
    else:
        reward = -1  # Penalty for incorrect character

    return reward


def one_hot_encode_password(password, char_space):
    """
    One-hot encodes a password string.

    Args:
        password: The password string.
        char_space: The character space.

    Returns:
        A NumPy array of one-hot encoded characters.
    """
    encoded_password = []
    for char in password:
        one_hot = np.zeros(len(char_space))
        if char in char_space:
            one_hot[char_space.index(char)] = 1
        encoded_password.append(one_hot)
    return np.array(encoded_password)

def create_char_space(passwords=None):
    """
    Creates a character space from a list of passwords or uses a default set.

    Args:
        passwords (optional): A list of passwords to generate the character space from.

    Returns:
        A string containing all unique characters.
    """
    if passwords:
        char_space = set()
        for password in passwords:
            char_space.update(password)
        return ''.join(char_space)
    else:
        # If no passwords are provided, use a default character set
        import string
        return string.ascii_letters + string.digits + string.punctuation