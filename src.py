

"""**RISEL Complete Pipeline**"""



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, average_precision_score,
                           confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# =========================
#  Data
# =========================
def load_and_preprocess_data(file_path: str):
    data = pd.read_csv(file_path)

    required_cols = ['Tem', 'Hum', 'Pressure', 'Rain', 'Light',
                     'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Moisture',
                     'Movements', 'Movements_t-1', 'Movements_t-2',
                     'Movements_t-3', 'Movements_t+1']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    movement_cols = ['Movements', 'Movements_t-1', 'Movements_t-2', 'Movements_t-3']
    data[movement_cols] = data[movement_cols].fillna(0)
    data = data.dropna()

    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])

    data = data.dropna(subset=['Movements_t+1'])
    X = data.drop(columns=['Movements_t+1'])
    y = data['Movements_t+1'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# =========================
#  Environment
# =========================
class LandslideEnv:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.current_idx = 0

    def reset(self, *, shuffle: bool = True):
        """Reset environment. shuffle=False keeps row order (needed for stacking)."""
        self.current_idx = 0
        if shuffle:
            indices = np.arange(len(self.X))
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]
        return self.X[self.current_idx]

    def step(self, action: int):
        true_label = self.y[self.current_idx]
        reward_matrix = {
            0: {0: +1, 1: -2, 2: -3},
            1: {0: -2, 1: +2, 2: -1},
            2: {0: -3, 1: -1, 2: +3}
        }
        reward = reward_matrix.get(true_label, {}).get(action, -1)
        self.current_idx += 1
        done = self.current_idx >= len(self.X) - 1
        next_state = self.X[self.current_idx] if not done else None
        return next_state, reward, done, true_label

# =========================
#  Neural Network Components
# =========================
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.fc(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.fc(x)

# =========================
# RL Model (Bse) Implementations
# =========================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, entropy_coeff=0.01):
        super().__init__()
        self.actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x): return self.actor(x)

    def train(self, states, actions, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        values = self.critic(states)
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        actor_loss = -(log_probs * advantages.detach().squeeze()).mean()
        critic_loss = nn.MSELoss()(values, rewards + self.gamma * next_values * (1 - dones))
        entropy = dist.entropy().mean()

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        (actor_loss - self.entropy_coeff * entropy + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()

class A2C(ActorCritic):
    def __init__(self, *args, n_steps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps

    def train(self, states, actions, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        values = self.critic(states)
        last_value = self.critic(states[-1:]).detach() if not dones[-1] else torch.zeros(1, 1).to(self.device)

        returns = torch.zeros_like(rewards)
        R = last_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        actor_loss = -(log_probs * advantages.detach().squeeze()).mean()
        critic_loss = nn.MSELoss()(values, returns.detach())
        entropy = dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()

class PPO(ActorCritic):
    def __init__(self, *args, clip_epsilon=0.2, ppo_epochs=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

    def train(self, states, actions, rewards, dones, old_log_probs):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        with torch.no_grad():
            values = self.critic(states)
            next_values = torch.cat([values[1:], torch.zeros(1, 1).to(self.device)])
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            s1 = ratios * advantages.detach().squeeze()
            s2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach().squeeze()
            actor_loss = -torch.min(s1, s2).mean()

            values = self.critic(states)
            critic_loss = nn.MSELoss()(values, returns)

            total_loss = actor_loss - self.entropy_coeff * entropy + 0.5 * critic_loss
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.action_dim = output_dim
        self.to(self.device)

    def forward(self, x): return self.fc(x)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.zeros_like(state) if next_state is None else np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0, self.epsilon
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.forward(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return loss.item(), self.epsilon

# =========================
#  Training and Evaluation with Cross-Validation
# =========================
def cross_validate_rl_model(model_class, X, y, n_splits=5, num_episodes=30, model_name="Model"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} for {model_name} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = model_class(X_train.shape[1], len(np.unique(y_train)))
        env_train = LandslideEnv(X_train, y_train)
        run_experiment(env_train, model, num_episodes=num_episodes, name=model_name)
        env_val = LandslideEnv(X_val, y_val)
        metrics = evaluate_model(env_val, model, model_name)
        fold_metrics.append(metrics)

    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics])
    }
    return avg_metrics

def run_experiment(env, model, num_episodes=30, name="Model"):
    print(f"\n Training {name}")
    best_f1 = -np.inf
    best_model_state = None
    no_improvement = 0
    early_stop_threshold = 30

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        true_labels, predictions = [], []

        if isinstance(model, DQN):
            states, actions, rewards, next_states, dones = [], [], [], [], []
        else:
            states, actions, rewards, dones, old_log_probs = [], [], [], [], []

        while not done:
            if isinstance(model, DQN):
                action = model.act(state)
                next_state, reward, done, true_label = env.step(action)
                model.remember(state, action, reward, next_state, done)
                loss, epsilon = model.replay()
                states.append(state); actions.append(action); rewards.append(reward)
                next_states.append(next_state if not done else None); dones.append(done)
            else:
                state_tensor = torch.FloatTensor(state).to(model.device)
                action_probs = model.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action).to(model.device)).item()
                next_state, reward, done, true_label = env.step(action)
                states.append(state); actions.append(action); rewards.append(reward)
                dones.append(done); old_log_probs.append(log_prob)

            state = next_state if not done else None
            episode_reward += reward
            true_labels.append(true_label)
            predictions.append(action)

        current_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= early_stop_threshold:
            print(f"Early stopping at episode {ep}")
            break

        if not isinstance(model, DQN):
            if isinstance(model, PPO):
                for _ in range(model.ppo_epochs):
                    actor_loss, critic_loss, entropy = model.train(states, actions, rewards, dones, old_log_probs)
            else:
                actor_loss, critic_loss, entropy = model.train(states, actions, rewards, dones)

        if isinstance(model, DQN):
            if ep % 10 == 0:
                print(f"Episode {ep}, Reward {episode_reward:.2f}, F1 {current_f1:.3f}, Loss {loss:.4f}, Eps {epsilon:.4f}")
        else:
            if ep % 10 == 0:
                print(f"Episode {ep}, Reward {episode_reward:.2f}, F1 {current_f1:.3f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f" {name} training complete")

def _compute_auc_from_probs(y_true, probs, classes):
    """Compute ROC/PR AUC with class-order aligned to probs columns."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    classes = np.asarray(classes)
    uniq = np.unique(y_true)

    mask = np.isin(classes, uniq)
    classes = classes[mask]
    probs = probs[:, mask]

    if len(classes) == 2:

        order = np.argsort(classes)
        classes = classes[order]
        probs = probs[:, order]
        yb = (y_true == classes[1]).astype(int)
        roc_auc = roc_auc_score(yb, probs[:, 1])
        pr_auc = average_precision_score(yb, probs[:, 1])
    else:
        yb = label_binarize(y_true, classes=classes)
        roc_auc = roc_auc_score(yb, probs, multi_class='ovr')
        pr_auc = average_precision_score(yb, probs, average='weighted')
    return roc_auc, pr_auc

def evaluate_model(env, model, model_name="Model"):
    """Comprehensive evaluation for a single RL model"""
    state = env.reset()
    preds, labels, probs = [], [], []
    done = False

    while not done:
        try:
            state_tensor = torch.FloatTensor(state).to(model.device if hasattr(model, "device") else "cpu")
            with torch.no_grad():
                if isinstance(model, DQN):
                    q_values = model(state_tensor).cpu().numpy()
                    prob = np.exp(q_values - np.max(q_values)) / np.sum(np.exp(q_values - np.max(q_values)))
                    action = int(np.argmax(q_values))
                else:
                    action_probs = model.actor(state_tensor).cpu().numpy()
                    prob = action_probs
                    action = int(np.argmax(action_probs))
            next_state, _, done, true_label = env.step(action)
            preds.append(action); labels.append(true_label); probs.append(prob)
            state = next_state if not done else None
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            break

    if not preds:
        print("Warning: No predictions during evaluation")
        return dict(accuracy=0,f1=0,precision=0,recall=0,roc_auc=0,pr_auc=0)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)

    # AUCs
    try:
        classes = np.arange(len(np.unique(labels)))  # RL actions indexed 0..K-1
        roc_auc, pr_auc = _compute_auc_from_probs(labels, probs, classes)
    except Exception as e:
        print(f"Couldn't calculate probability metrics: {str(e)}")
        roc_auc, pr_auc = 0, 0

    cm = confusion_matrix(labels, preds)
    print(f"\n {model_name} Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout(); plt.show()

    return {'accuracy': accuracy,'f1': f1,'precision': precision,'recall': recall,'roc_auc': roc_auc,'pr_auc': pr_auc}

# =========================
#  Stacked Ensemble
# =========================
class StackedEnsembleRL:
    def __init__(self, base_models, meta_models=None):
        self.base_models = base_models
        self.meta_models = meta_models if meta_models else self._initialize_default_meta_models()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.device = torch.device("cpu")
        self.trained_meta_models = {}

    def _initialize_default_meta_models(self):
        return [
            ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=42)),
            ('LightGBM', LGBMClassifier(random_state=42)),
            ('CatBoost', CatBoostClassifier(verbose=0, random_state=42)),
            ('Logistic_Lasso', LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)),
            ('GradientBoosting', GradientBoostingClassifier(random_state=42))
        ]


    def _get_base_model_predictions(self, X: np.ndarray, model: nn.Module):
        preds, probs = [], []
        for i in range(len(X)):
            x = torch.FloatTensor(X[i]).to(self.device)
            with torch.no_grad():
                if isinstance(model, DQN):
                    q = model(x).cpu().numpy()
                    p = np.exp(q - np.max(q)) / np.sum(np.exp(q - np.max(q)))
                    a = int(np.argmax(q))
                else:
                    p = model.actor(x).cpu().numpy()
                    a = int(np.argmax(p))
            preds.append(a); probs.append(p)
        return np.asarray(preds), np.asarray(probs)

    def _evaluate_meta_model(self, model, X_train, y_train, X_val, y_val):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_val, preds),
            'f1': f1_score(y_val, preds, average='weighted', zero_division=0),
            'precision': precision_score(y_val, preds, average='weighted', zero_division=0),
            'recall': recall_score(y_val, preds, average='weighted', zero_division=0)
        }

        if probs is not None:
            try:

                classes = getattr(model, "classes_", np.unique(y_train))
                roc_auc, pr_auc = _compute_auc_from_probs(y_val, probs, classes)
                metrics['roc_auc'] = roc_auc
                metrics['pr_auc']  = pr_auc
            except Exception as e:
                print(f"    Couldn't calculate probability metrics: {str(e)}")
                metrics['roc_auc'] = 0; metrics['pr_auc'] = 0

        return metrics, preds, probs

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        results = {
            'base_models_cv': {},
            'base_models_train': {},
            'base_models_test': {},
            'meta_models_cv': {},
            'meta_models_train': {},
            'meta_models_test': {}
        }

        # === Base models on training set (report only)
        print("\nEvaluating base models on full training set...")
        for i, model in enumerate(self.base_models):
            preds, probs = self._get_base_model_predictions(X_train, model)
            metrics = {
                'accuracy': accuracy_score(y_train[:len(preds)], preds),
                'f1': f1_score(y_train[:len(preds)], preds, average='weighted', zero_division=0),
                'precision': precision_score(y_train[:len(preds)], preds, average='weighted', zero_division=0),
                'recall': recall_score(y_train[:len(preds)], preds, average='weighted', zero_division=0)
            }
            # RL base models don't expose classes; use action indices 0..K-1
            try:
                classes = np.arange(probs.shape[1])
                roc_auc, pr_auc = _compute_auc_from_probs(y_train[:len(preds)], probs, classes)
            except Exception:
                roc_auc, pr_auc = 0, 0
            metrics['roc_auc'] = roc_auc; metrics['pr_auc'] = pr_auc
            results['base_models_train'][f'BaseModel_{i}'] = metrics

        # === Base models on test set (report only)
        print("\nEvaluating base models on test set...")
        for i, model in enumerate(self.base_models):
            preds, probs = self._get_base_model_predictions(X_test, model)
            metrics = {
                'accuracy': accuracy_score(y_test[:len(preds)], preds),
                'f1': f1_score(y_test[:len(preds)], preds, average='weighted', zero_division=0),
                'precision': precision_score(y_test[:len(preds)], preds, average='weighted', zero_division=0),
                'recall': recall_score(y_test[:len(preds)], preds, average='weighted', zero_division=0)
            }
            try:
                classes = np.arange(probs.shape[1])
                roc_auc, pr_auc = _compute_auc_from_probs(y_test[:len(preds)], probs, classes)
            except Exception:
                roc_auc, pr_auc = 0, 0
            metrics['roc_auc'] = roc_auc; metrics['pr_auc'] = pr_auc
            results['base_models_test'][f'BaseModel_{i}'] = metrics

        # === Meta features for CV (no leakage) ===
        print("\nPreparing meta features for cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_metrics = {name: [] for name, _ in self.meta_models}

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nProcessing fold {fold + 1}/5")
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            # Build meta features deterministically for train & val
            tr_preds, tr_probs = [], []
            va_preds, va_probs = [], []
            min_train_len, min_val_len = float('inf'), float('inf')

            for model in self.base_models:
                p_tr, pr_tr = self._get_base_model_predictions(X_tr, model)
                p_va, pr_va = self._get_base_model_predictions(X_va, model)
                tr_preds.append(p_tr); tr_probs.append(pr_tr); min_train_len = min(min_train_len, len(p_tr))
                va_preds.append(p_va); va_probs.append(pr_va); min_val_len = min(min_val_len, len(p_va))

            X_meta_tr = np.column_stack([p[:min_train_len] for p in tr_preds] +
                                        [pr[:min_train_len] for pr in [pp.reshape(min_train_len, -1) for pp in tr_probs]])
            y_meta_tr = y_tr[:min_train_len]

            X_meta_va = np.column_stack([p[:min_val_len] for p in va_preds] +
                                        [pr[:min_val_len] for pr in [pp.reshape(min_val_len, -1) for pp in va_probs]])
            y_meta_va = y_va[:min_val_len]

            # Impute/scale by TRAIN fold only
            X_meta_tr = self.imputer.fit_transform(X_meta_tr)
            X_meta_tr = self.scaler.fit_transform(X_meta_tr)
            X_meta_va = self.imputer.transform(X_meta_va)
            X_meta_va = self.scaler.transform(X_meta_va)

            for name, model in self.meta_models:
                try:
                    metrics, _, _ = self._evaluate_meta_model(copy.deepcopy(model), X_meta_tr, y_meta_tr, X_meta_va, y_meta_va)
                    cv_metrics[name].append(metrics)
                except Exception as e:
                    print(f"Error with {name} in fold {fold + 1}: {str(e)}")

        for name in cv_metrics:
            if cv_metrics[name]:
                results['meta_models_cv'][name] = {
                    'accuracy': np.mean([m['accuracy'] for m in cv_metrics[name]]),
                    'f1': np.mean([m['f1'] for m in cv_metrics[name]]),
                    'precision': np.mean([m['precision'] for m in cv_metrics[name]]),
                    'recall': np.mean([m['recall'] for m in cv_metrics[name]]),
                    'roc_auc': np.mean([m.get('roc_auc', 0) for m in cv_metrics[name]]),
                    'pr_auc': np.mean([m.get('pr_auc', 0) for m in cv_metrics[name]])
                }

        # === Final meta features for train/test ===
        print("\nPreparing final meta features...")
        train_preds, train_probs, min_train = [], [], float('inf')
        test_preds, test_probs, min_test = [], [], float('inf')

        for model in self.base_models:
            p_tr, pr_tr = self._get_base_model_predictions(X_train, model)
            train_preds.append(p_tr); train_probs.append(pr_tr); min_train = min(min_train, len(p_tr))
            p_te, pr_te = self._get_base_model_predictions(X_test, model)
            test_preds.append(p_te); test_probs.append(pr_te); min_test = min(min_test, len(p_te))

        X_meta_train = np.column_stack([p[:min_train] for p in train_preds] +
                                       [pr[:min_train] for pr in [pp.reshape(min_train, -1) for pp in train_probs]])
        y_meta_train = y_train[:min_train]

        X_meta_test = np.column_stack([p[:min_test] for p in test_preds] +
                                      [pr[:min_test] for pr in [pp.reshape(min_test, -1) for pp in test_probs]])
        y_meta_test = y_test[:min_test]

        X_meta_train = self.imputer.fit_transform(X_meta_train)
        X_meta_train = self.scaler.fit_transform(X_meta_train)
        X_meta_test = self.imputer.transform(X_meta_test)
        X_meta_test = self.scaler.transform(X_meta_test)

        # === Train meta models on full meta-train; evaluate on meta-test ===
        print("\nTraining meta models on full training set...")
        for name, model in self.meta_models:
            try:
                metrics, preds, probs = self._evaluate_meta_model(copy.deepcopy(model),
                                                                  X_meta_train, y_meta_train,
                                                                  X_meta_train, y_meta_train)
                results['meta_models_train'][name] = metrics
                model.fit(X_meta_train, y_meta_train)
                self.trained_meta_models[name] = model

                cm = confusion_matrix(y_meta_train, preds)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"Meta Model {name}\nTraining Confusion Matrix")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.tight_layout(); plt.show()
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results['meta_models_train'][name] = None

        print("\nEvaluating meta models on test set...")
        for name, model in self.trained_meta_models.items():
            try:
                # Evaluate properly on held-out meta-test
                preds = model.predict(X_meta_test)
                metrics = {
                    'accuracy': accuracy_score(y_meta_test, preds),
                    'f1': f1_score(y_meta_test, preds, average='weighted', zero_division=0),
                    'precision': precision_score(y_meta_test, preds, average='weighted', zero_division=0),
                    'recall': recall_score(y_meta_test, preds, average='weighted', zero_division=0)
                }
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_meta_test)
                    classes = getattr(model, "classes_", np.unique(y_meta_train))
                    roc_auc, pr_auc = _compute_auc_from_probs(y_meta_test, probs, classes)
                    metrics['roc_auc'] = roc_auc; metrics['pr_auc'] = pr_auc
                else:
                    metrics['roc_auc'] = 0; metrics['pr_auc'] = 0

                results['meta_models_test'][name] = metrics

                cm = confusion_matrix(y_meta_test, preds)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.title(f"Meta Model {name}\nTest Confusion Matrix")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.tight_layout(); plt.show()
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results['meta_models_test'][name] = None

        return results

# =========================
#  Main Execution
# =========================
def main():
    try:
        print("Loading data...")
        X, y = load_and_preprocess_data("GriffonPeak5_t.csv")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()

        print("\nTraining base RL models with cross-validation...")
        rl_models = {'ACM': ActorCritic, 'A2C': A2C, 'PPO': PPO, 'DQN': DQN}
        base_model_metrics = {}
        trained_models = []

        for name, model_class in rl_models.items():
            print(f"\n=== Training {name} ===")
            cv_metrics = cross_validate_rl_model(model_class, X_train, y_train, n_splits=3, num_episodes=100, model_name=name)
            print(f"\nTraining final {name} model on full training data...")
            final_model = model_class(X_train.shape[1], len(np.unique(y_train)))
            env_train = LandslideEnv(X_train, y_train)
            run_experiment(env_train, final_model, num_episodes=100, name=f"Final {name}")
            trained_models.append(final_model)

            print(f"\nEvaluating {name} on training set...")
            env_train = LandslideEnv(X_train, y_train)
            train_metrics = evaluate_model(env_train, final_model, model_name=f"Train {name}")

            print(f"\nEvaluating {name} on test set...")
            env_test = LandslideEnv(X_test, y_test)
            test_metrics = evaluate_model(env_test, final_model, model_name=f"Test {name}")

            base_model_metrics[name] = {'cv': cv_metrics, 'train': train_metrics, 'test': test_metrics}

        print("\nBuilding stacked ensemble...")
        ensemble = StackedEnsembleRL(base_models=trained_models)
        ensemble_results = ensemble.fit_and_evaluate(X_train, y_train, X_test, y_test)

        print("\n=== Final Results ===")
        print("\n=== Base Models ===")
        for name, metrics in base_model_metrics.items():
            print(f"\n{name}:")
            print("  Cross-Validation Metrics:")
            for metric, value in metrics['cv'].items():
                print(f"    {metric:>10}: {value:.4f}")
            print("  Training Set Metrics:")
            for metric, value in metrics['train'].items():
                print(f"    {metric:>10}: {value:.4f}")
            print("  Test Set Metrics:")
            for metric, value in metrics['test'].items():
                print(f"    {metric:>10}: {value:.4f}")

        print("\n=== Ensemble Models ===")
        print("\n  Cross-Validation Metrics:")
        for name, metrics in ensemble_results['meta_models_cv'].items():
            if metrics:
                print(f"\n    {name}:")
                for metric, value in metrics.items():
                    print(f"      {metric:>10}: {value:.4f}")

        print("\n  Training Set Metrics:")
        for name, metrics in ensemble_results['meta_models_train'].items():
            if metrics:
                print(f"\n    {name}:")
                for metric, value in metrics.items():
                    print(f"      {metric:>10}: {value:.4f}")

        print("\n  Test Set Metrics:")
        for name, metrics in ensemble_results['meta_models_test'].items():
            if metrics:
                print(f"\n    {name}:")
                for metric, value in metrics.items():
                    print(f"      {metric:>10}: {value:.4f}")

        # Comparison plot
        all_metrics = {}
        for name, metrics in base_model_metrics.items():
            all_metrics[f"{name} (CV)"] = metrics['cv']
            all_metrics[f"{name} (Train)"] = metrics['train']
            all_metrics[f"{name} (Test)"] = metrics['test']
        for name, metrics in ensemble_results['meta_models_cv'].items():
            if metrics: all_metrics[f"Ensemble {name} (CV)"] = metrics
        for name, metrics in ensemble_results['meta_models_train'].items():
            if metrics: all_metrics[f"Ensemble {name} (Train)"] = metrics
        for name, metrics in ensemble_results['meta_models_test'].items():
            if metrics: all_metrics[f"Ensemble {name} (Test)"] = metrics

        df_metrics = pd.DataFrame(all_metrics).T
        print("\nPerformance Comparison:")
        print(df_metrics)

        metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc']
        plt.figure(figsize=(16, 10))
        df_metrics[metrics_to_plot].plot(kind='bar', rot=45)
        plt.title("Model Performance Comparison")
        plt.ylabel("Score"); plt.ylim(0, 1); plt.grid(True, axis='y')
        plt.tight_layout(); plt.show()

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback; traceback.print_exc()
        print("Please check your data and parameters.")

if __name__ == "__main__":
    main()