import numpy as np

class VarianceEarlyStopping:
    def __init__(self, window_size=10, patience=5, min_epochs=0, delta=1e-6):
        """
        Args:
            window_size (int): Number of recent loss values to compute one variance.
            patience (int): Number of times the mean variance can fail to decrease before stopping.
            min_epochs (int): Minimum number of epochs before applying early stopping.
            delta (float): Minimum change in mean variance to count as a decrease.
        """
        self.window_size = window_size
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta

        self.loss_buffer = []
        self.variance_history = []
        self.output_buffer = []
        self.wait = 0
        self.best_mean_variance = float('inf')
        self.epoch_count = 0
        self.should_stop = False

    def update(self, loss_value, output):
        """
        Args:
            loss_value (float): The current loss (e.g., per epoch).
        
        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        self.epoch_count += 1
        self.loss_buffer.append(loss_value)
        self.output_buffer.append(output)

        # Maintain buffer size
        if len(self.loss_buffer) < self.window_size:
            return False  # Not enough data yet

        if len(self.loss_buffer) > self.window_size:
            self.loss_buffer.pop(0)
            self.output_buffer.pop(0)

        # Compute variance over current window
        var = np.var(self.loss_buffer)
        self.variance_history.append(var)

        # Check mean trend of variance
        if self.epoch_count >= self.min_epochs:
            current_mean_var = np.mean(self.variance_history)
            print(f"Epoch {self.epoch_count}: Current Mean Variance = {current_mean_var:.8f}")
            if self.best_mean_variance - current_mean_var > self.delta:
                # Significant decrease
                self.best_mean_variance = current_mean_var
                self.wait = 0
            else:
                # No meaningful decrease
                self.wait += 1

            if self.wait >= self.patience:
                self.should_stop = True

        return self.should_stop

    def get_flag(self):
        """
        Returns the current status of early stopping.

        Returns:
            bool: True if early stopping should trigger, else False.
        """
        return self.should_stop

    def get_images(self):
        """
        Returns the outputs collected so far.

        Returns:
            list: List of outputs collected during training.
        """
        return self.output_buffer
    
    def get_losses(self):
        """
        Returns the losses collected so far.

        Returns:
            list: List of losses collected during training.
        """
        return self.loss_buffer


class MeanEarlyStopping:
    def __init__(self, window_size=10, patience=5, min_epochs=0, delta=1e-6):
        """
        Args:
            window_size (int): Number of recent loss values to compute one variance.
            patience (int): Number of times the mean variance can fail to decrease before stopping.
            min_epochs (int): Minimum number of epochs before applying early stopping.
            delta (float): Minimum change in mean variance to count as a decrease.
        """
        self.window_size = window_size
        self.patience = patience
        self.min_epochs = min_epochs
        self.delta = delta

        self.loss_buffer = []
        self.variance_history = []
        self.output_buffer = []
        self.wait = 0
        self.best_mean_variance = float('inf')
        self.epoch_count = 0
        self.should_stop = False

    def update(self, loss_value, output):
        """
        Args:
            loss_value (float): The current loss (e.g., per epoch).
        
        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        self.epoch_count += 1
        self.loss_buffer.append(loss_value)
        self.output_buffer.append(output)

        # Maintain buffer size
        if len(self.loss_buffer) < self.window_size:
            return False  # Not enough data yet

        if len(self.loss_buffer) > self.window_size:
            self.loss_buffer.pop(0)
            self.output_buffer.pop(0)

        # Compute mean over current window
        var = np.mean(self.loss_buffer)
        self.variance_history.append(var)

        # Check mean trend of mean
        if self.epoch_count >= self.min_epochs:
            current_mean_var = np.mean(self.variance_history)
            print(f"Epoch {self.epoch_count}: Current Avg Mean = {current_mean_var:.8f}")
            if self.best_mean_variance - current_mean_var > self.delta:
                # Significant decrease
                self.best_mean_variance = current_mean_var
                self.wait = 0
            else:
                # No meaningful decrease
                self.wait += 1

            if self.wait >= self.patience:
                self.should_stop = True

        return self.should_stop

    def get_flag(self):
        """
        Returns the current status of early stopping.

        Returns:
            bool: True if early stopping should trigger, else False.
        """
        return self.should_stop

    def get_images(self):
        """
        Returns the outputs collected so far.

        Returns:
            list: List of outputs collected during training.
        """
        return self.output_buffer
    
    def get_losses(self):
        """
        Returns the losses collected so far.

        Returns:
            list: List of losses collected during training.
        """
        return self.loss_buffer
