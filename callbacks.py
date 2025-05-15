import tensorflow as tf
import numpy as np

class CurriculumLearningCallback(tf.keras.callbacks.Callback):
    """Callback for curriculum learning strategy.
    
    Gradually increases KL weight during training.
    """
    
    def __init__(self, total_steps, final_kl_weight):
        super().__init__()
        self.total_steps = total_steps
        self.final_kl_weight = final_kl_weight
        self.step = 0
        
    def on_batch_end(self, batch, logs=None):
        self.step += 1
        progress = min(self.step / self.total_steps, 1.0)
        self.model.kl_weight.assign(progress * self.final_kl_weight)
