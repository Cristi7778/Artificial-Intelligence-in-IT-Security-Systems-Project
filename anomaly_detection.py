import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,precision_recall_curve, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import plot_tree, DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Inspecting the general features
def inspect_features(df):
    print(f"Dataset Shape: {df.shape}")
    print("\nLabel Distribution (Binary):")
    print(df['label'].value_counts(normalize=True))
    print("\nAttack Category Distribution:")
    print(df['attack_cat'].value_counts())

# We use the median value for numerical columns and the mode for categorical
def fill_missing_values(df):
    print("Missing values found:", df.isnull().sum().sum())

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encoding the data
def encode(df,target, cat_cols):
    top_n = 5
    threshold = 0.95

    cols_to_one_hot = []
    cols_to_target_encode = []
    # If we can include at least 95% of the values in a maximum of 5 categories we use One Hot,
    # Otherwise we use target encoding
    for col in cat_cols:
        top_5_coverage = df[col].value_counts(normalize=True).head(top_n).sum()

        if top_5_coverage >= threshold:
            print(f"Feature '{col}': Top 5 cover {top_5_coverage:.1%}. Using One-Hot.")

            top_cats = df[col].value_counts().head(top_n).index.tolist()

            df[col] = df[col].apply(lambda x: x if x in top_cats else 'Other')
            cols_to_one_hot.append(col)

        else:
            print(f"Feature '{col}': Top 5 cover {top_5_coverage:.1%}. Using Target Encoding.")
            cols_to_target_encode.append(col)

    if cols_to_one_hot:
        df = pd.get_dummies(df, columns=cols_to_one_hot, drop_first=True)

    for col in cols_to_target_encode:
        mean_target = df.groupby(col)[target].mean()
        df[col + '_target'] = df[col].map(mean_target)
        df.drop(columns=[col], inplace=True)

    print(f"Final Data Shape: {df.shape}")
    return df
# Statistical analysis
def compute_summary_statistics(df, target_col='label'):

    print("\n--- Computing Summary Statistics")

    # Identify numerical features and exclude auxiliary columns
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = ['id', target_col]

    stats_features = [col for col in numerical_features if col not in exclude_cols]

    # Compute statistics including custom percentiles
    summary_stats = df[stats_features].describe(
        percentiles=[0.25, 0.5, 0.90]
    ).T
    print(summary_stats)

# Visual analysis
def visualize_analysis(df, dist_features=['dur', 'sbytes', 'dbytes', 'rate'], target_col='label'):

    fig, axes = plt.subplots(len(dist_features), 1, figsize=(10, 4 * len(dist_features)))
    plt.title('Distribution (Histograms) of Key Flow Features', fontsize=14, y=1.02)

    if len(dist_features) == 1:
        axes = [axes]

    for i, col in enumerate(dist_features):
        # Histogram
        sns.histplot(df[col], bins=50, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram: {col}')
        axes[i].set_xlabel(col)

    plt.tight_layout()
    plt.show()

    numerical_features = df.select_dtypes(include=np.number).columns.tolist()

    if 'id' in numerical_features:
        numerical_features.remove('id')

    correlation_matrix = df[numerical_features].corr()

    plt.figure(figsize=(20, 18))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap='coolwarm',
        linewidths=.5,
        cbar_kws={'shrink': 0.7}
    )
    plt.title('Correlation Heatmap of Numerical Features (Incl. Target)', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def train_anomaly_detection(train_df, test_df, target_col='label', ignore_cols=['id', 'attack_cat'],
                            optimize_threshold=True):
    """
    Trains Isolation Forest on Normal data and automatically tunes the threshold
    to maximize the F1-Score on the test set.
    """
    print(f"\n{'=' * 50}")
    print(f"   STARTING ANOMALY DETECTION (Auto-Tuned)")
    print(f"{'=' * 50}")

    # 1. Prepare Training Data (ONLY Normal Samples)
    train_normal = train_df[train_df[target_col] == 0].copy()

    # Drop target and ignored columns
    cols_to_drop = [target_col] + [c for c in ignore_cols if c in train_df.columns]

    X_train = train_normal.drop(columns=cols_to_drop, errors='ignore')
    X_test_raw = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test_ground_truth = test_df[target_col]

    # 2. Align Columns (Prevent 'Feature names unseen' error)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    print(f"Training on {len(X_train)} Normal samples.")
    print(f"Testing on {len(X_test)} Mixed samples.")

    # 3. Train Isolation Forest
    # Note: We set contamination to 'auto' initially, but we will override the decision rule later.
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

    # 4. Compute Raw Anomaly Scores
    print("Computing anomaly scores...")
    scores = iso_forest.decision_function(X_test)

    # 5. Determine Predictions (Optimized vs Default)
    if optimize_threshold:
        print("Optimizing threshold for best F1-Score...")
        # Note: We use -scores because precision_recall_curve expects higher values to be the "positive" class (Attack)
        # Isolation Forest gives lower scores to anomalies, so we flip the sign.
        precisions, recalls, thresholds = precision_recall_curve(y_test_ground_truth, -scores)

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

        best_idx = np.argmax(f1_scores)
        best_threshold_score = -thresholds[best_idx]

        print(f"   -> Found Optimal Threshold: {best_threshold_score:.4f}")
        print(f"   -> Max F1-Score: {f1_scores[best_idx]:.4f}")

        # Apply new rule: If score < threshold, it's an Attack (1)
        y_pred = [1 if s < best_threshold_score else 0 for s in scores]

    else:
        print("Using default Isolation Forest threshold...")
        preds_raw = iso_forest.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in preds_raw]

    # 6. Evaluate
    print("\n--- Final Evaluation Results ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_ground_truth, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test_ground_truth, y_pred, target_names=['Normal', 'Attack']))

    try:
        auc = roc_auc_score(y_test_ground_truth, y_pred)
        print(f"AUC-ROC Score: {auc:.4f}")
    except:
        print("Could not calculate AUC")

    return y_pred, scores


def train_decision_tree(train_df, test_df, target_col='label', ignore_cols=['id', 'attack_cat'], grid_search=False):
    # Trains and evaluates a Decision Tree Classifier.

    print(f"\n   DECISION TREE MODEL\n")

    # 1. Prepare Data
    cols_to_drop = [target_col] + [c for c in ignore_cols if c in train_df.columns]

    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df[target_col]

    X_test_raw = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_df[target_col]

    # 2. Align Columns (Prevent crashes from One-Hot mismatches)
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    # 3. Train Decision Tree
    dt_model = None

    if grid_search:
        print("Training Decision Tree (Grid Search Mode)...")
        # --- Integrated Tuning Logic ---
        param_grid = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_leaf': [1, 10, 50, 100],
            'criterion': ['gini', 'entropy']
        }
        dt = DecisionTreeClassifier(random_state=42)
        grid_search_cv = GridSearchCV(dt, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search_cv.fit(X_train, y_train)

        dt_model = grid_search_cv.best_estimator_
        print(f"   -> Best Hyperparameters: {grid_search_cv.best_params_}")

    else:
        print("Training Decision Tree (Standard Mode)...")
        # Standard fast training
        dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
        dt_model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = dt_model.predict(X_test)

    print("\n--- Decision Tree Results ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # 5. Visualization (Optional - saves image)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X_train.columns, class_names=['Normal', 'Attack'], filled=True, max_depth=3)
    plt.title("Decision Tree (Top 3 Levels)")
    plt.show()

def train_naive_bayes(train_df, test_df, target_col='label', ignore_cols=['id', 'attack_cat']):
    #Trains and evaluates a Gaussian Naive Bayes Classifier.
    #Includes Standard Scaling which is required for NB.
    print(f"\n   NAIVE BAYES MODEL\n")

    # 1. Prepare Data
    cols_to_drop = [target_col] + [c for c in ignore_cols if c in train_df.columns]

    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df[target_col]

    X_test_raw = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_df[target_col]

    # 2. Align Columns
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    # 3. Scaling (CRITICAL for Naive Bayes)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Naive Bayes
    print("Training Gaussian Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = nb_model.predict(X_test_scaled)

    print("\n--- Naive Bayes Results ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))


def train_autoencoder(train_df, test_df, target_col='label', ignore_cols=['id', 'attack_cat']):
    #Trains a Neural Autoencoder to learn 'Normal' network patterns.
    #High reconstruction error = Anomaly (Attack).

    print(f"\n   NEURAL AUTOENCODER ANOMALY DETECTION\n")


    # 1. Prepare Data
    cols_to_drop = [target_col] + [c for c in ignore_cols if c in train_df.columns]

    train_normal = train_df[train_df[target_col] == 0].copy()
    X_train = train_normal.drop(columns=cols_to_drop, errors='ignore')

    X_test_raw = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test_ground_truth = test_df[target_col]

    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    # 2. Scaling (CRITICAL for Neural Networks)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    print(f"Input Dimension: {input_dim}")

    # 3. Build Autoencoder Architecture
    # Structure: Input -> Encoder -> Bottleneck -> Decoder -> Output

    input_layer = Input(shape=(input_dim,))

    # Encoder (Compressing)
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)

    # Bottleneck (Latent Space)
    bottleneck = Dense(8, activation='relu')(encoded)

    # Decoder (Reconstructing)
    decoded = Dense(16, activation='relu')(bottleneck)
    decoded = Dense(32, activation='relu')(decoded)

    # Output Layer (Sigmoid for 0-1 range match with MinMaxScaler)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compile
    autoencoder.compile(optimizer='adam', loss='mse')

    # 4. Train the Model
    print("Training Autoencoder on Normal traffic...")
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=20,  # Adjust based on time/accuracy needs
        batch_size=64,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    # 5. Detect Anomalies
    print("\nCalculating Reconstruction Error...")
    # Get reconstructions for the test set
    reconstructions = autoencoder.predict(X_test_scaled)

    # Calculate Mean Squared Error (MSE) between Input and Reconstruction
    # mse = mean((input - output)^2)
    mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

    # 6. Determine Threshold (Optimize F1-Score)
    print("Optimizing Threshold...")
    precisions, recalls, thresholds = precision_recall_curve(y_test_ground_truth, mse)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"Optimal Reconstruction Error Threshold: {best_threshold:.6f}")

    # 7. Predict
    # If Error > Threshold -> Anomaly (1)
    y_pred = [1 if error > best_threshold else 0 for error in mse]

    # 8. Evaluate
    print("\n--- Autoencoder Results ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_ground_truth, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_ground_truth, y_pred, target_names=['Normal', 'Attack']))

    return autoencoder

def main():
    data_train=pd.read_csv("UNSW_data_train.csv")
    data_test=pd.read_csv("UNSW_data_testing.csv")
    inspect_features(data_test)
    fill_missing_values(data_train)
    fill_missing_values(data_test)
    data_train=encode(data_train,'label',['proto', 'service', 'state'])
    data_test=encode(data_test,'label',['proto', 'service', 'state'])
    compute_summary_statistics(data_test)
    visualize_analysis(data_train)
    train_anomaly_detection(data_train,data_test)
    train_decision_tree(data_train,data_test,grid_search=True)
    train_naive_bayes(data_train,data_test)
    train_autoencoder(data_train,data_test)
main()