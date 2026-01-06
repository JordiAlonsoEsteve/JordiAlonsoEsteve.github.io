import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from log_barrier_svm import phase_I, phase_II
from primal_dual_svm import primal_dual_SVM
from SMO import SMO
from data_utils import construct_Q_rbf

# -----------------------------
# Helper function to compute accuracy for dual alpha
# -----------------------------
def test_alphas_with_bias(X_train, y_train, X_eval, y_eval, alpha, C, gamma=0.05):
    # RBF kernel for training
    X1_sq = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X_train**2, axis=1).reshape(1, -1)
    dist_sq_train = X1_sq + X2_sq - 2 * X_train @ X_train.T
    K_train = np.exp(-gamma * dist_sq_train)

    # Identify support vectors
    sv_mask = (alpha > -1e4) & (alpha < (C + 1e-4))
    if not np.any(sv_mask):
        import pdb; pdb.set_trace()
        raise ValueError("No support vectors found. Check alpha values or C parameter.")

    # Compute bias term b
    b_values = y_train[sv_mask] - (alpha * y_train) @ K_train[:, sv_mask]
    b = np.mean(b_values)

    # Compute kernel between training and evaluation points
    X1_sq_eval = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X2_sq_eval = np.sum(X_eval**2, axis=1).reshape(1, -1)
    dist_sq_eval = X1_sq_eval + X2_sq_eval - 2 * X_train @ X_eval.T
    K_eval = np.exp(-gamma * dist_sq_eval)

    y_pred_scores = (alpha * y_train) @ K_eval + b
    y_pred = np.sign(y_pred_scores)

    accuracy = np.mean(y_pred == y_eval)
    return accuracy

# -----------------------------
# Load dataset: digits 3 vs 5
# -----------------------------
X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
target = y

mask = (target == '4') | (target == '6')
data = X[mask]
target = target[mask]
# Map 3 -> -1, 5 -> +1
target = np.array([-1 if x == '4' else 1 for x in target])
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, shuffle=True, random_state=42
)
print(np.mean(y_test == 1))
print(X_train.shape)
print(X_test.shape)

# 2. Initialize the StandardScaler
scaler = MinMaxScaler()

# 3. Fit on training data AND transform it
# This calculates the mean and std dev for each feature based ONLY on X_train
X_train = scaler.fit_transform(X_train)

# 4. Transform the test data
# We use the mean and std dev from the training set to ensure consistency
X_test = scaler.transform(X_test)

C = 0.1
results = {}

# -----------------------------
# 1. primal_dual_SVM
# -----------------------------
start = time.time()
alpha, lambd, nu, alphas_list, value_function, nu_list = primal_dual_SVM(
    X_train, y_train, C, Q_function=construct_Q_rbf, t = 100.0
)
end = time.time()
train_acc = test_alphas_with_bias(X_train, y_train, X_train, y_train, alpha, C)
test_acc = test_alphas_with_bias(X_train, y_train, X_test, y_test, alpha, C)
results['primal_dual_SVM'] = {'time': end-start, 'train_acc': train_acc, 'test_acc': test_acc}

# -----------------------------
# 2. phase_I + phase_II
# -----------------------------
start = time.time()
alpha_init = np.zeros_like(y_train, dtype=float)
t, s, nu_val = 100.0, 100, 1.0
alpha_phase, _, nu_val = phase_I(alpha_init, y_train, C, s=s, t=t, nu=nu_val)
alpha_phase, alphas_list, value_function, nu_list = phase_II(
    x=X_train, y=y_train, C=C, alpha= alpha_phase.copy(), nu=nu_val, Q_function=construct_Q_rbf
)
end = time.time()
train_acc = test_alphas_with_bias(X_train, y_train, X_train, y_train, alpha_phase, C)
test_acc = test_alphas_with_bias(X_train, y_train, X_test, y_test, alpha_phase, C)

results['Log-Barrier'] = {'time': end-start, 'train_acc': train_acc, 'test_acc': test_acc}
# -----------------------------
# 3. SMO
# -----------------------------
start = time.time()
alpha_smo, b_smo, alphas_list_smo, history_smo = SMO(
    X_train, y_train, max_iter=500, C=C, track_progress=False
)
end = time.time()
train_acc = test_alphas_with_bias(X_train, y_train, X_train, y_train, alpha_smo, C)
test_acc = test_alphas_with_bias(X_train, y_train, X_test, y_test, alpha_smo, C)
results['SMO'] = {'time': end-start, 'train_acc': train_acc, 'test_acc': test_acc}

# -----------------------------
# 4. sklearn SVC
# -----------------------------
start = time.time()
clf = SVC(kernel='rbf', C=C)
clf.fit(X_train, y_train)
end = time.time()
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
results['sklearn_SVC'] = {'time': end-start, 'train_acc': train_acc, 'test_acc': test_acc}

# -----------------------------
# Report results
# -----------------------------
print("Comparison of SVM implementations on 3 vs 5 digits:\n")
print(f"{'Method':15s} | {'Time (s)':>8s} | {'Train Acc':>10s} | {'Test Acc':>9s}")
print("-"*50)
for name, res in results.items():
    print(f"{name:15s} | {res['time']:8.4f} | {res['train_acc']*100:9.2f}% | {res['test_acc']*100:8.2f}%")

# -----------------------------
# Bar plots
# -----------------------------
methods = list(results.keys())
times = [res['time'] for res in results.values()]
train_accs = [res['train_acc'] for res in results.values()]
test_accs = [res['test_acc'] for res in results.values()]

label_for_3 = -1 
label_for_5 = 1

plt.figure(figsize=(12, 10))

# --- Performance Metrics ---
plt.subplot(2, 2, 1)
plt.bar(methods, times, color='skyblue')
plt.ylabel("Time (s)")
plt.title("Computation Time")

plt.subplot(2, 2, 2)
x = np.arange(len(methods))
width = 0.35
plt.bar(x - width/2, train_accs, width, label='Train Acc', color='lightgreen')
plt.bar(x + width/2, test_accs, width, label='Test Acc', color='salmon')
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.xticks(x, methods)
plt.title("Training vs Test Accuracy")
plt.legend()

# --- Digit Visualization ---
idx_3 = np.where(y_test == label_for_3)[0][40]
idx_5 = np.where(y_test == label_for_5)[0][40]

# Reshape to 28x28 (standard MNIST dimensions)
img_3 = X_test[idx_3].reshape(16, 16)
img_5 = X_test[idx_5].reshape(16, 16)

plt.subplot(2, 2, 3)
plt.imshow(img_3, cmap='Greys')
plt.title(f"Sample Digit (Label: {label_for_3})")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_5, cmap='Greys')
plt.title(f"Sample Digit (Label: {label_for_5})")
plt.axis('off')

plt.tight_layout()
plt.savefig("Comparison.png")
plt.show()