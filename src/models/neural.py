"""Custom classification and regression models for EnergyTypeNet."""

import inspect
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.kernel_approximation import RBFSampler

class ActivationFunctions:
    """Common neural-network activation functions and derivatives."""

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_prime(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z: np.ndarray) -> np.ndarray:
        s = ActivationFunctions.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_prime(z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def leaky_relu(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 0.01 * z)

    @staticmethod
    def leaky_relu_prime(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.01)

    @staticmethod
    def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))

    @staticmethod
    def elu_prime(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(z > 0, 1.0, alpha * np.exp(z))

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_prime(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    @staticmethod
    def plot_all(ax=None):
        """Plot activation functions and derivatives."""
        import matplotlib.pyplot as plt

        z = np.linspace(-5, 5, 400)
        functions = [
            ('relu', ActivationFunctions.relu, ActivationFunctions.relu_prime),
            ('sigmoid', ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime),
            ('tanh', ActivationFunctions.tanh, ActivationFunctions.tanh_prime),
            ('leaky_relu', ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_prime),
            ('elu', ActivationFunctions.elu, ActivationFunctions.elu_prime),
            ('linear', ActivationFunctions.linear, ActivationFunctions.linear_prime),
        ]

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        else:
            axes = ax
            fig = axes[0].figure

        for name, func, deriv in functions:
            axes[0].plot(z, func(z), label=name)
            axes[1].plot(z, deriv(z), label=f'{name} prime')

        axes[0].set_title('Activation functions')
        axes[1].set_title('Activation derivatives')
        axes[1].set_xlabel('z')
        for axis in axes:
            axis.axhline(0, color='black', linewidth=0.7)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=3)

        return fig

class MLPCustom(BaseEstimator):
    """Multi-layer perceptron implemented from scratch with NumPy."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = 'relu',
        output_activation: str = 'softmax',
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 32,
        optimizer: str = 'sgd',
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_init: str = 'he',
        random_scale: float = 0.01,
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        validation_fraction: float = 0.0,
        patience: int = 20,
        random_state: int | None = None,
        task: str = 'classification',
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_init = weight_init
        self.random_scale = random_scale
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.task = task
        self.verbose = verbose

    def __repr__(self) -> str:
        return (
            f'MLPCustom(hidden_layer_sizes={self.hidden_layer_sizes}, '
            f'activation={self.activation!r}, optimizer={self.optimizer!r}, '
            f'task={self.task!r})'
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPCustom':
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._rng = np.random.RandomState(self.random_state)

        X_train, y_train, X_val, y_val = self._validation_split(X, y)
        y_fit = self._prepare_target(y_train)
        self._initialize_parameters(X_train.shape[1], y_fit.shape[1])

        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_epochs_ = 0
        best_val = np.inf
        wait = 0
        best_state = None

        for epoch in range(self.n_iterations):
            indices = self._rng.permutation(X_train.shape[0])
            X_epoch = X_train[indices]
            y_epoch = y_fit[indices]

            for start in range(0, X_epoch.shape[0], self.batch_size):
                stop = start + self.batch_size
                X_batch = X_epoch[start:stop]
                y_batch = y_epoch[start:stop]
                cache = self._forward(X_batch, training=True)
                grads_w, grads_b = self._backward(cache, y_batch)
                self._update_parameters(grads_w, grads_b)

            train_loss = self._loss(self._forward(X_train, training=False)['A'][-1], y_fit)
            self.loss_history_.append(train_loss)
            self.n_epochs_ = epoch + 1

            if X_val is not None:
                y_val_fit = self._encode_existing_target(y_val)
                val_loss = self._loss(self._forward(X_val, training=False)['A'][-1], y_val_fit)
                self.val_loss_history_.append(val_loss)

                if val_loss < best_val - 1e-8:
                    best_val = val_loss
                    wait = 0
                    best_state = ([w.copy() for w in self.weights_], [b.copy() for b in self.biases_])
                else:
                    wait += 1
                    if wait >= self.patience:
                        if best_state is not None:
                            self.weights_, self.biases_ = best_state
                        break

            if self.verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f'epoch={epoch + 1}, loss={train_loss:.5f}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = self._forward(np.asarray(X, dtype=float), training=False)['A'][-1]

        if self.task == 'classification':
            return self.classes_[np.argmax(output, axis=1)]

        return output.ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != 'classification':
            raise ValueError('predict_proba is only available for classification')

        return self._forward(np.asarray(X, dtype=float), training=False)['A'][-1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y)
        pred = self.predict(X)

        if self.task == 'classification':
            return float(np.mean(pred == y))

        ss_res = np.sum((y.ravel() - pred.ravel()) ** 2)
        ss_tot = np.sum((y.ravel() - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _validation_split(self, X: np.ndarray, y: np.ndarray):
        if self.validation_fraction <= 0:
            return X, y, None, None

        n_val = max(1, int(round(X.shape[0] * self.validation_fraction)))
        indices = self._rng.permutation(X.shape[0])
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            self.classes_, encoded = np.unique(y, return_inverse=True)
            one_hot = np.zeros((encoded.size, self.classes_.size))
            one_hot[np.arange(encoded.size), encoded] = 1.0
            self.n_outputs_ = self.classes_.size
            return one_hot

        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.n_outputs_ = y.shape[1]
        return y

    def _encode_existing_target(self, y: np.ndarray) -> np.ndarray:
        if self.task == 'classification':
            encoded = np.searchsorted(self.classes_, y)
            one_hot = np.zeros((encoded.size, self.classes_.size))
            one_hot[np.arange(encoded.size), encoded] = 1.0
            return one_hot

        return np.asarray(y, dtype=float).reshape(-1, 1)

    def _initialize_parameters(self, n_features: int, n_outputs: int) -> None:
        if not hasattr(self, '_rng'):
            self._rng = np.random.RandomState(self.random_state)

        layer_sizes = [n_features, *list(self.hidden_layer_sizes), n_outputs]
        self.weights_ = []
        self.biases_ = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if self.weight_init == 'he':
                scale = np.sqrt(2.0 / n_in)
            elif self.weight_init == 'xavier':
                scale = np.sqrt(2.0 / (n_in + n_out))
            elif self.weight_init == 'random':
                scale = self.random_scale
            else:
                raise ValueError('weight_init must be he, xavier, or random')

            self.weights_.append(self._rng.normal(0.0, scale, size=(n_in, n_out)))
            self.biases_.append(np.zeros((1, n_out)))

        self._velocity_w = [np.zeros_like(w) for w in self.weights_]
        self._velocity_b = [np.zeros_like(b) for b in self.biases_]
        self._adam_m_w = [np.zeros_like(w) for w in self.weights_]
        self._adam_v_w = [np.zeros_like(w) for w in self.weights_]
        self._adam_m_b = [np.zeros_like(b) for b in self.biases_]
        self._adam_v_b = [np.zeros_like(b) for b in self.biases_]
        self._step = 0

    def _forward(self, X: np.ndarray, training: bool = False) -> dict[str, list[np.ndarray]]:
        activations = [X]
        pre_activations = []
        dropout_masks = []
        A = X

        for layer_idx, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            Z = A @ W + b
            pre_activations.append(Z)

            is_output = layer_idx == len(self.weights_) - 1
            if is_output:
                A = self._output_activation(Z)
                dropout_masks.append(np.ones_like(A))
            else:
                A = self._hidden_activation(Z)
                if training and self.dropout_rate > 0:
                    keep_prob = 1.0 - self.dropout_rate
                    mask = (self._rng.rand(*A.shape) < keep_prob) / keep_prob
                    A = A * mask
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(np.ones_like(A))

            activations.append(A)

        return {'Z': pre_activations, 'A': activations, 'dropout_masks': dropout_masks}

    def _backward(self, cache: dict[str, list[np.ndarray]], y_true: np.ndarray):
        A = cache['A']
        Z = cache['Z']
        n_samples = y_true.shape[0]

        if self.task == 'classification':
            dZ = (A[-1] - y_true) / n_samples
        else:
            dZ = (2.0 * (A[-1] - y_true) / n_samples) * self._output_prime(Z[-1])

        grads_w = [None] * len(self.weights_)
        grads_b = [None] * len(self.biases_)

        for layer_idx in reversed(range(len(self.weights_))):
            grads_w[layer_idx] = A[layer_idx].T @ dZ + self.l2_lambda * self.weights_[layer_idx]
            grads_b[layer_idx] = np.sum(dZ, axis=0, keepdims=True)

            if layer_idx > 0:
                dA_prev = dZ @ self.weights_[layer_idx].T
                dA_prev *= cache['dropout_masks'][layer_idx - 1]
                dZ = dA_prev * self._hidden_prime(Z[layer_idx - 1])

        return grads_w, grads_b

    def _update_parameters(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        self._step += 1

        for i in range(len(self.weights_)):
            if self.optimizer == 'sgd':
                self.weights_[i] -= self.learning_rate * grads_w[i]
                self.biases_[i] -= self.learning_rate * grads_b[i]
            elif self.optimizer == 'sgd_momentum':
                self._velocity_w[i] = self.momentum * self._velocity_w[i] - self.learning_rate * grads_w[i]
                self._velocity_b[i] = self.momentum * self._velocity_b[i] - self.learning_rate * grads_b[i]
                self.weights_[i] += self._velocity_w[i]
                self.biases_[i] += self._velocity_b[i]
            elif self.optimizer == 'adam':
                self._adam_m_w[i] = self.beta1 * self._adam_m_w[i] + (1.0 - self.beta1) * grads_w[i]
                self._adam_v_w[i] = self.beta2 * self._adam_v_w[i] + (1.0 - self.beta2) * (grads_w[i] ** 2)
                self._adam_m_b[i] = self.beta1 * self._adam_m_b[i] + (1.0 - self.beta1) * grads_b[i]
                self._adam_v_b[i] = self.beta2 * self._adam_v_b[i] + (1.0 - self.beta2) * (grads_b[i] ** 2)

                m_w_hat = self._adam_m_w[i] / (1.0 - self.beta1 ** self._step)
                v_w_hat = self._adam_v_w[i] / (1.0 - self.beta2 ** self._step)
                m_b_hat = self._adam_m_b[i] / (1.0 - self.beta1 ** self._step)
                v_b_hat = self._adam_v_b[i] / (1.0 - self.beta2 ** self._step)

                self.weights_[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)
                self.biases_[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)
            else:
                raise ValueError('optimizer must be sgd, sgd_momentum, or adam')

    def _loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.task == 'classification':
            clipped = np.clip(y_pred, 1e-12, 1.0)
            data_loss = -np.mean(np.sum(y_true * np.log(clipped), axis=1))
        else:
            data_loss = np.mean((y_pred - y_true) ** 2)

        l2_loss = 0.5 * self.l2_lambda * sum(np.sum(w ** 2) for w in self.weights_)
        return float(data_loss + l2_loss)

    def _hidden_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return ActivationFunctions.relu(z)
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid(z)
        if self.activation == 'tanh':
            return ActivationFunctions.tanh(z)
        if self.activation == 'leaky_relu':
            return ActivationFunctions.leaky_relu(z)
        raise ValueError('activation must be relu, sigmoid, tanh, or leaky_relu')

    def _hidden_prime(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return ActivationFunctions.relu_prime(z)
        if self.activation == 'sigmoid':
            return ActivationFunctions.sigmoid_prime(z)
        if self.activation == 'tanh':
            return ActivationFunctions.tanh_prime(z)
        if self.activation == 'leaky_relu':
            return ActivationFunctions.leaky_relu_prime(z)
        raise ValueError('activation must be relu, sigmoid, tanh, or leaky_relu')

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        if self.output_activation == 'softmax':
            return ActivationFunctions.softmax(z)
        if self.output_activation == 'linear':
            return ActivationFunctions.linear(z)
        raise ValueError('output_activation must be softmax or linear')

    def _output_prime(self, z: np.ndarray) -> np.ndarray:
        if self.output_activation == 'linear':
            return ActivationFunctions.linear_prime(z)
        return np.ones_like(z)
