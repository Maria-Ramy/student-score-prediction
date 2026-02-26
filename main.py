import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================

print("=" * 60)
print("STUDENT SCORE PREDICTION - DATA ANALYSIS")
print("=" * 60)

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")
print(f"\n📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# Basic info
print("\n📋 Dataset Info:")
print(df.info())

# Check missing values
print("\n🔍 Missing Values:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_percent})
print(missing_df[missing_df['Missing'] > 0])

# ============================================
# PART 2: DATA CLEANING (FIXED VERSION)
# ============================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Create a copy to work with
df_clean = df.copy()

# Handle missing values - FIXED: separate numerical and categorical columns
numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns

print(f"\n📊 Found {len(numerical_cols)} numerical columns and {len(categorical_cols)} categorical columns")

# Fill missing values for numerical columns with median
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"   - Filled {col} with median")

# Fill missing values for categorical columns with mode
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        print(f"   - Filled {col} with mode")

print("\n✅ Missing values handled")
print("\nVerifying no missing values remain:")
print(df_clean.isnull().sum().sum())  # Should print 0

# ============================================
# PART 3: EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Statistical summary
print("\n📊 Statistical Summary:")
print(df_clean.describe())

# Correlation matrix for numerical features
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df_clean[numerical_cols].corr()

# Figure 1: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100)
plt.show()

# Figure 2: Distribution of Exam Scores
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['Exam_Score'], bins=30, kde=True, color='#2E86AB')
plt.axvline(df_clean['Exam_Score'].mean(), color='red', linestyle='--',
            label=f"Mean: {df_clean['Exam_Score'].mean():.2f}")
plt.axvline(df_clean['Exam_Score'].median(), color='green', linestyle='--',
            label=f"Median: {df_clean['Exam_Score'].median():.2f}")
plt.xlabel('Exam Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Exam Scores', fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('score_distribution.png', dpi=100)
plt.show()

# Figure 3: Hours Studied vs Exam Score
plt.figure(figsize=(10, 6))
plt.scatter(df_clean['Hours_Studied'], df_clean['Exam_Score'], alpha=0.5, c='#A23B72', edgecolors='black',
            linewidth=0.5)
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.title('Hours Studied vs Exam Score', fontsize=16, fontweight='bold')
z = np.polyfit(df_clean['Hours_Studied'], df_clean['Exam_Score'], 1)
p = np.poly1d(z)
plt.plot(df_clean['Hours_Studied'].sort_values(), p(df_clean['Hours_Studied'].sort_values()),
         "r--", linewidth=2, label=f'Trend line (slope: {z[0]:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig('hours_vs_score.png', dpi=100)
plt.show()

# Figure 4: Box plots for categorical features vs score
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
                        'Internet_Access', 'School_Type', 'Gender']

for i, feature in enumerate(categorical_features):
    row, col = i // 3, i % 3
    df_clean.boxplot(column='Exam_Score', by=feature, ax=axes[row, col])
    axes[row, col].set_title(f'Exam Score by {feature}')
    axes[row, col].set_xlabel('')
    axes[row, col].tick_params(axis='x', rotation=45)

plt.suptitle('Impact of Categorical Features on Exam Scores', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('categorical_impact.png', dpi=100)
plt.show()

# ============================================
# PART 4: FEATURE ENGINEERING
# ============================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Create new features
df_clean['Study_Efficiency'] = df_clean['Exam_Score'] / (df_clean['Hours_Studied'] + 1)  # Avoid division by zero
df_clean['Sleep_Study_Ratio'] = df_clean['Sleep_Hours'] / (df_clean['Hours_Studied'] + 1)
df_clean['Attendance_Quality'] = df_clean['Attendance'] * df_clean['Previous_Scores'] / 100

print("✅ Created new features:")
print("   - Study_Efficiency: Score per hour studied")
print("   - Sleep_Study_Ratio: Sleep hours relative to study hours")
print("   - Attendance_Quality: Attendance × Previous Scores")

# ============================================
# PART 5: DATA PREPARATION FOR MODELING
# ============================================

print("\n" + "=" * 60)
print("DATA PREPARATION FOR MODELING")
print("=" * 60)

# Separate features and target
X = df_clean.drop('Exam_Score', axis=1)
y = df_clean['Exam_Score']

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"🔢 Numerical features: {len(numerical_features)}")
print(f"🏷️ Categorical features: {len(categorical_features)}")

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()

for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

print("✅ Categorical variables encoded")

# Scale numerical features
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

print("✅ Numerical features standardized")

# ============================================
# PART 6: MODEL 1 - SIMPLE LINEAR REGRESSION
# ============================================

print("\n" + "=" * 60)
print("MODEL 1: SIMPLE LINEAR REGRESSION (Hours_Studied only)")
print("=" * 60)

X_simple = X_encoded[['Hours_Studied']].copy()
y_simple = y.copy()

# Split data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Train model
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

# Predict
y_pred_s = simple_model.predict(X_test_s)

# Evaluate
mse_s = mean_squared_error(y_test_s, y_pred_s)
rmse_s = np.sqrt(mse_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print(f"📈 Model Performance:")
print(f"   - MSE: {mse_s:.4f}")
print(f"   - RMSE: {rmse_s:.4f}")
print(f"   - MAE: {mae_s:.4f}")
print(f"   - R² Score: {r2_s:.4f}")
print(f"\n📐 Equation: Exam_Score = {simple_model.coef_[0]:.4f} × Hours_Studied + {simple_model.intercept_:.4f}")

# Figure 5: Simple Linear Regression Results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter with regression line
axes[0].scatter(X_test_s, y_test_s, alpha=0.6, color='#2E86AB', label='Actual')
X_test_s_sorted = X_test_s.sort_values(by='Hours_Studied')
axes[0].plot(X_test_s_sorted, simple_model.predict(X_test_s_sorted),
             color='#A23B72', linewidth=2, label='Predicted')
axes[0].set_xlabel('Hours Studied (standardized)', fontsize=12)
axes[0].set_ylabel('Exam Score', fontsize=12)
axes[0].set_title('Simple Linear Regression', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals plot
residuals_s = y_test_s - y_pred_s
axes[1].scatter(y_pred_s, residuals_s, alpha=0.6, color='#F18F01')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Predicted Values', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_linear_results.png', dpi=100)
plt.show()

# ============================================
# PART 7: MODEL 2 - MULTIPLE LINEAR REGRESSION
# ============================================

print("\n" + "=" * 60)
print("MODEL 2: MULTIPLE LINEAR REGRESSION")
print("=" * 60)

# Select features - make sure this list matches what we use in prediction
selected_features = ['Hours_Studied', 'Attendance', 'Previous_Scores',
                     'Tutoring_Sessions', 'Parental_Involvement',
                     'Access_to_Resources', 'Motivation_Level', 'Study_Efficiency']
X_multi = X_encoded[selected_features].copy()
y_multi = y.copy()

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train model
multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# Predict
y_pred_m = multi_model.predict(X_test_m)

# Evaluate
mse_m = mean_squared_error(y_test_m, y_pred_m)
rmse_m = np.sqrt(mse_m)
mae_m = mean_absolute_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print(f"📈 Model Performance:")
print(f"   - MSE: {mse_m:.4f}")
print(f"   - RMSE: {rmse_m:.4f}")
print(f"   - MAE: {mae_m:.4f}")
print(f"   - R² Score: {r2_m:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': multi_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n🔍 Feature Coefficients:")
print(feature_importance.to_string(index=False))

# Figure 6: Feature Importance
plt.figure(figsize=(10, 6))
colors = ['#2E86AB' if x > 0 else '#A23B72' for x in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance in Multiple Linear Regression', fontsize=16, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()

# ============================================
# PART 8: MODEL 3 - POLYNOMIAL REGRESSION
# ============================================

print("\n" + "=" * 60)
print("MODEL 3: POLYNOMIAL REGRESSION")
print("=" * 60)

# Try different polynomial degrees
degrees = [2, 3]
poly_results = {}

for degree in degrees:
    # Create pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

    # Train
    poly_pipeline.fit(X_train_m, y_train_m)

    # Predict
    y_pred_p = poly_pipeline.predict(X_test_m)

    # Evaluate
    mse_p = mean_squared_error(y_test_m, y_pred_p)
    rmse_p = np.sqrt(mse_p)
    r2_p = r2_score(y_test_m, y_pred_p)

    poly_results[degree] = {'mse': mse_p, 'rmse': rmse_p, 'r2': r2_p, 'model': poly_pipeline}

    print(f"\n📊 Degree {degree} Polynomial:")
    print(f"   - MSE: {mse_p:.4f}")
    print(f"   - RMSE: {rmse_p:.4f}")
    print(f"   - R² Score: {r2_p:.4f}")

# Select best polynomial model
best_degree = min(poly_results, key=lambda x: poly_results[x]['mse'])
best_poly_model = poly_results[best_degree]['model']
y_pred_poly = best_poly_model.predict(X_test_m)

# ============================================
# PART 9: MODEL 4 - REGULARIZED REGRESSION
# ============================================

print("\n" + "=" * 60)
print("MODEL 4: REGULARIZED REGRESSION")
print("=" * 60)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_m, y_train_m)
y_pred_ridge = ridge_model.predict(X_test_m)
r2_ridge = r2_score(y_test_m, y_pred_ridge)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_m, y_train_m)
y_pred_lasso = lasso_model.predict(X_test_m)
r2_lasso = r2_score(y_test_m, y_pred_lasso)

print(f"📈 Ridge Regression R²: {r2_ridge:.4f}")
print(f"📈 Lasso Regression R²: {r2_lasso:.4f}")

# ============================================
# PART 10: MODEL COMPARISON
# ============================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Simple Linear', 'Multiple Linear', f'Polynomial (deg={best_degree})', 'Ridge', 'Lasso'],
    'MSE': [mse_s, mse_m, poly_results[best_degree]['mse'],
            mean_squared_error(y_test_m, y_pred_ridge),
            mean_squared_error(y_test_m, y_pred_lasso)],
    'RMSE': [rmse_s, rmse_m, poly_results[best_degree]['rmse'],
             np.sqrt(mean_squared_error(y_test_m, y_pred_ridge)),
             np.sqrt(mean_squared_error(y_test_m, y_pred_lasso))],
    'R²': [r2_s, r2_m, poly_results[best_degree]['r2'], r2_ridge, r2_lasso]
})

print("\n📊 Model Performance Comparison:")
print(comparison.round(4).to_string(index=False))

# Figure 7: Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# R² Comparison
bars1 = axes[0].bar(comparison['Model'], comparison['R²'],
                    color=['#2E86AB', '#A23B72', '#F18F01', '#3B8F5E', '#D95D39'])
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 1)
for bar, val in zip(bars1, comparison['R²']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# RMSE Comparison
bars2 = axes[1].bar(comparison['Model'], comparison['RMSE'],
                    color=['#2E86AB', '#A23B72', '#F18F01', '#3B8F5E', '#D95D39'])
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('RMSE Comparison (lower is better)', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, comparison['RMSE']):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100)
plt.show()

# Figure 8: Predicted vs Actual (Best Model)
best_model_name = comparison.loc[comparison['R²'].idxmax(), 'Model']
best_r2 = comparison['R²'].max()

plt.figure(figsize=(8, 8))
plt.scatter(y_test_m, y_pred_m, alpha=0.6, color='#2E86AB', edgecolors='black', linewidth=0.5)
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Exam Score', fontsize=12)
plt.ylabel('Predicted Exam Score', fontsize=12)
plt.title(f'Best Model: {best_model_name}\nR² = {best_r2:.4f}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('best_model_predictions.png', dpi=100)
plt.show()

# ============================================
# PART 11: CROSS-VALIDATION
# ============================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION")
print("=" * 60)

# Perform cross-validation on the best model
cv_scores = cross_val_score(multi_model, X_multi, y_multi, cv=5, scoring='r2')
print(f"📊 5-Fold Cross-Validation R² Scores: {cv_scores}")
print(f"   - Mean: {cv_scores.mean():.4f}")
print(f"   - Std: {cv_scores.std():.4f}")

# ============================================
# PART 12: SIMPLE PREDICTION FUNCTION (FIXED)
# ============================================

print("\n" + "=" * 60)
print("PREDICTION FUNCTION")
print("=" * 60)


def predict_student_score_simple(hours_studied, attendance, previous_scores,
                                 tutoring_sessions, parental_involvement,
                                 access_to_resources, motivation_level):
    """
    Simple prediction using just the selected features
    """
    # Create a simple dataframe with only the features we need
    input_data = pd.DataFrame({
        'Hours_Studied': [hours_studied],
        'Attendance': [attendance],
        'Previous_Scores': [previous_scores],
        'Tutoring_Sessions': [tutoring_sessions],
        'Parental_Involvement': [parental_involvement],
        'Access_to_Resources': [access_to_resources],
        'Motivation_Level': [motivation_level],
        'Study_Efficiency': [previous_scores / (hours_studied + 1)]
    })

    # Create a temporary dataframe for encoding
    input_encoded = input_data.copy()

    # Encode categorical variables
    for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level']:
        if col in label_encoders:
            # Handle unseen labels
            try:
                input_encoded[col] = label_encoders[col].transform(input_encoded[col].astype(str))
            except:
                # If value not in encoder, use the most common class (0)
                input_encoded[col] = 0

    # Scale ONLY the numerical columns
    numerical_to_scale = ['Hours_Studied', 'Attendance', 'Previous_Scores',
                          'Tutoring_Sessions', 'Study_Efficiency']

    # Scale each column individually
    for col in numerical_to_scale:
        # Get the scaler's mean and std for this column
        col_idx = numerical_features.index(col) if col in numerical_features else -1
        if col_idx >= 0:
            mean_val = scaler.mean_[col_idx]
            std_val = np.sqrt(scaler.var_[col_idx])
            input_encoded[col] = (input_encoded[col] - mean_val) / std_val

    # Predict
    prediction = multi_model.predict(input_encoded[selected_features])

    return prediction[0]


# Test the prediction function
print("\n🧪 Testing prediction function with sample values...")
try:
    test_prediction = predict_student_score_simple(
        hours_studied=20,
        attendance=85,
        previous_scores=75,
        tutoring_sessions=2,
        parental_involvement='Medium',
        access_to_resources='High',
        motivation_level='High'
    )

    print(f"\n🎯 Sample Prediction:")
    print(f"   Student: Hours=20, Attendance=85%, Previous=75, Tutoring=2")
    print(f"   Predicted Exam Score: {test_prediction:.2f}")

    # Test a few more examples
    print("\n🧪 Testing more examples:")
    test_cases = [
        (10, 70, 60, 0, 'Low', 'Low', 'Low'),
        (25, 95, 90, 3, 'High', 'High', 'High'),
        (15, 80, 70, 1, 'Medium', 'Medium', 'Medium')
    ]

    for hours, att, prev, tutor, parent, access, motive in test_cases:
        pred = predict_student_score_simple(hours, att, prev, tutor, parent, access, motive)
        print(f"   Hours={hours}, Att={att}%, Prev={prev}, Tutor={tutor} → Predicted: {pred:.2f}")

except Exception as e:
    print(f"⚠️ Error in prediction function: {e}")
    print("   Using fallback method...")

    # Fallback: use the model coefficients directly
    intercept = multi_model.intercept_
    coef_dict = dict(zip(selected_features, multi_model.coef_))


    # Manual calculation
    def manual_predict(hours, att, prev, tutor, parent_val, access_val, motive_val):
        # Encode categories (simple mapping)
        cat_map = {'Low': 0, 'Medium': 1, 'High': 2}
        parent = cat_map.get(parent_val, 1)
        access = cat_map.get(access_val, 1)
        motive = cat_map.get(motive_val, 1)
        efficiency = prev / (hours + 1)

        pred = intercept
        pred += coef_dict['Hours_Studied'] * hours
        pred += coef_dict['Attendance'] * att
        pred += coef_dict['Previous_Scores'] * prev
        pred += coef_dict['Tutoring_Sessions'] * tutor
        pred += coef_dict['Parental_Involvement'] * parent
        pred += coef_dict['Access_to_Resources'] * access
        pred += coef_dict['Motivation_Level'] * motive
        pred += coef_dict['Study_Efficiency'] * efficiency
        return pred


    print("\n🎯 Sample Prediction (Fallback):")
    pred = manual_predict(20, 85, 75, 2, 'Medium', 'High', 'High')
    print(f"   Predicted Exam Score: {pred:.2f}")

# ============================================
# PART 13: SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save comparison to CSV
comparison.to_csv('model_comparison_results.csv', index=False)
print("✅ Model comparison saved to 'model_comparison_results.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance saved to 'feature_importance.csv'")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated files:")
print("   - correlation_heatmap.png")
print("   - score_distribution.png")
print("   - hours_vs_score.png")
print("   - categorical_impact.png")
print("   - simple_linear_results.png")
print("   - feature_importance.png")
print("   - model_comparison.png")
print("   - best_model_predictions.png")
print("   - model_comparison_results.csv")
print("   - feature_importance.csv")

# ============================================
# PART 13: SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save comparison to CSV
comparison.to_csv('model_comparison_results.csv', index=False)
print("✅ Model comparison saved to 'model_comparison_results.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ Feature importance saved to 'feature_importance.csv'")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated files:")
print("   - correlation_heatmap.png")
print("   - score_distribution.png")
print("   - hours_vs_score.png")
print("   - categorical_impact.png")
print("   - simple_linear_results.png")
print("   - feature_importance.png")
print("   - model_comparison.png")
print("   - best_model_predictions.png")
print("   - model_comparison_results.csv")
print("   - feature_importance.csv")