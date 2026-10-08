# Logistic regression: the gradient of the loss (T04)

One example: logit z = w·x + b, probability p = σ(z) = 1 / (1 + e^(−z)), label y ∈ {0, 1},
loss L = −[y·log p + (1 − y)·log(1 − p)] (binary cross-entropy, BCE).

Chain rule: dL/dw = dL/dp · dp/dz · dz/dw.

1. dL/dp = −y/p + (1 − y)/(1 − p) = (p − y) / (p(1 − p)).
2. dp/dz = σ(z)(1 − σ(z)) = p(1 − p)   (the sigmoid's derivative).
3. Multiply: dL/dz = (p − y) / (p(1 − p)) · p(1 − p) = **p − y**.
4. dz/dw = x and dz/db = 1, so dL/dw = (p − y)·x and dL/db = p − y.
5. Mean over a batch of n rows (X: (n, d), p and y: (n,)):
   **dL/dw = Xᵀ(p − y) / n**, dL/db = mean(p − y).

Why the form is so simple: the p(1 − p) from the sigmoid cancels the p(1 − p) in the
denominator of the BCE derivative. The error signal is "prediction minus label", the same as in
linear regression with squared error. With squared error instead of BCE, the factor p(1 − p)
would stay, and it is close to 0 when the neuron is confidently wrong (p ≈ 0 or 1), so learning
would stall exactly when the mistake is biggest. BCE is the loss that "undoes" the sigmoid.

Checked numerically in `tests/test_logreg.py::test_gradient_check_float64` (relative error < 1e-6).
