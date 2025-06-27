import numpy as np

class VarianceEarlyStopping:
    def __init__(self, window_size=10, var_thresh=1e-4, min_epochs=0):
        """
        Args:
            window_size (int): Number of recent loss values to track.
            var_thresh (float): Threshold for variance below which to stop.
            min_epochs (int): Minimum number of epochs before applying early stopping.
        """
        self.window_size = window_size
        self.var_thresh = var_thresh
        self.min_epochs = min_epochs
        self.loss_buffer = []
        self.epoch_count = 0
        self.stop_flag = False

    def update(self, loss_value):
        """
        Call this after each epoch or iteration.

        Args:
            loss_value (float): The current epoch's or iteration's loss.
        
        Returns:
            bool: True if early stopping should trigger, else False.
        """
        self.epoch_count += 1
        self.loss_buffer.append(loss_value)

        # Maintain buffer size
        if len(self.loss_buffer) > self.window_size:
            self.loss_buffer.pop(0)

        # Check stop condition
        if (
            self.epoch_count >= self.min_epochs and
            len(self.loss_buffer) == self.window_size
        ):
            var = np.var(self.loss_buffer)
            if var < self.var_thresh:
                self.stop_flag = True

        return self.stop_flag
