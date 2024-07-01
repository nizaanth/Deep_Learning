import tensorflow.keras.backend as K


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """Implementation of Focal Loss.
    Equation:
        loss = -alpha*((1-p)^gamma)*log(p)

    Args:
        gamma (float, optional): the same as wighting factor
          in balanced cross entropy. Defaults to 2.0.
        alpha (float, optional): focusing parameter for
          modulating factor (1-p). Defaults to 0.25.
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()        
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of a modulating factor and weighting factor
        weight = alpha * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=-1))
        return loss
    
    return focal_loss