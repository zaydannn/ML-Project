# Import all necessary librarires
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
import warnings
import pickle

# ignores unnecessary warnings
warnings.filterwarnings('ignore')

# Streamlit Page configuration
st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
# These keep trackk of what steps have been completed and stored
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar navigation
# These enable the user to select which page to visit
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["📊 Data Import & Overview", "🔧 Data Preprocessing", "🤖 Model Training",
     "📈 Model Evaluation", "🔮 Prediction", "📋 Conclusions"]
)


# Helper functions
@st.cache_data
def load_data():
    """Load the loan dataset"""
    try:
        # Load the provided CSV data
        data = pd.read_csv('loan.csv')
        return data
    except:
        # Sends error warning if file not found
        st.error("Please upload the loan.csv file to proceed.")
        return None


def preprocess_data(df):
    """Preprocess the loan dataset"""
    df = df.copy()

    # Handle missing values with either mode(for categorical columns) or median(numerical columns)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

    # Create new features for Feature Engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_LoanAmount_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1)
    df['LoanAmount_Log'] = np.log(df['LoanAmount'] + 1)

    # Encode categorical variables into numerical values
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education',
                           'Self_Employed', 'Property_Area']

    for col in categorical_columns:
        df[col + '_Encoded'] = le.fit_transform(df[col])

    # Encode target variable
    df['Loan_Status_Encoded'] = le.fit_transform(df['Loan_Status'])

    return df


# PAGE 1: Data Import and Overview
if page == "📊 Data Import & Overview": #Checks to see if the user has currently selected page 1:
    st.markdown('<div class="main-header">🏦 Loan Default Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Data Import and Overview</div>', unsafe_allow_html=True)

    # File upload option to upload dataset
    uploaded_file = st.file_uploader("Upload loan dataset (CSV)", type=['csv'])

    # Checks if file was actually uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) #if a file was actually uploaded, then it should be read.
        st.session_state.df_original = df
        st.session_state.data_loaded = True
    else:
        # Try to load the provided data
        df = load_data()
        if df is not None:
            st.session_state.df_original = df
            st.session_state.data_loaded = True

    if st.session_state.data_loaded:
        df = st.session_state.df_original

        # Dataset overview
        # Creates a success box to show that data haas been successfully loaded
        st.markdown('<div class="success-box">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)

        # creates columns to show data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            approved_loans = len(df[df['Loan_Status'] == 'Y'])
            st.metric("Approved Loans", approved_loans)
        with col4:
            approval_rate = (approved_loans / len(df)) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")

        # Display dataset sample
        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        # Data types and missing values
        # Creates 2 columns for EDA
        col1, col2 = st.columns(2)

        # Column 1 to display summary of the data
        with col1:
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count()
            })
            st.dataframe(dtype_df, use_container_width=True)

        # Column 2 for missing value analysis
        with col2:
            st.markdown("**Missing Values**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found!")

        # Summary statistics
        st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        # VISUALIZATIONS
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        # Loan approval distribution in a pie chart
        fig1 = px.pie(df, names='Loan_Status', title='Loan Approval Distribution',
                      color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'})
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)

        # Creates 2 columns side by side
        col1, col2 = st.columns(2)
        # Income distribution in a histogram with plotly for interactivity
        with col1:
            fig2 = px.histogram(df, x='ApplicantIncome', title='Applicant Income Distribution',
                                nbins=30, color_discrete_sequence=['#3498db'])
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Loan amount distribution in a histogram with plotly for interactivity
        with col2:
            fig3 = px.histogram(df, x='LoanAmount', title='Loan Amount Distribution',
                                nbins=30, color_discrete_sequence=['#9b59b6'])
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # Correlation matrix in a heatmap
        st.markdown("**Correlation Matrix (Numerical Features)**")
        corr_matrix = df[numeric_cols].corr()
        fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         title="Correlation Matrix", color_continuous_scale='RdBu')
        st.plotly_chart(fig4, use_container_width=True)

        # Categorical variable analysis and bar charts for categorical distribution
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        for i in range(0, len(categorical_cols), 2):
            col1, col2 = st.columns(2)

            with col1:
                if i < len(categorical_cols):
                    fig = px.bar(df[categorical_cols[i]].value_counts().reset_index(),
                                 x='count', y=categorical_cols[i],
                                 title=f'{categorical_cols[i]} Distribution')
                    fig.update_layout(xaxis_title=categorical_cols[i], yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if i + 1 < len(categorical_cols):
                    fig = px.bar(df[categorical_cols[i + 1]].value_counts().reset_index(),
                                 x='count', y=categorical_cols[i + 1],
                                 title=f'{categorical_cols[i + 1]} Distribution')
                    fig.update_layout(xaxis_title=categorical_cols[i + 1], yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: Data Preprocessing
elif page == "🔧 Data Preprocessing": # checks if user has selected page 2
    st.markdown('<div class="main-header">Data Preprocessing</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded: #checks if user has loaded the dataset
        st.warning("Please load the dataset first from the 'Data Import & Overview' page.") # sends a warning message if user has not loaded his dataset
    else:
        df = st.session_state.df_original.copy()

        st.markdown('<div class="section-header">Missing Values Analysis</div>', unsafe_allow_html=True)

        # Show missing values and percentage of missing values on the data before preprocessing
        missing_before = df.isnull().sum()
        missing_before_df = pd.DataFrame({
            'Column': missing_before.index,
            'Missing Count': missing_before.values,
            'Missing %': (missing_before.values / len(df)) * 100
        })
        missing_before_df = missing_before_df[missing_before_df['Missing Count'] > 0]

        if len(missing_before_df) > 0:
            st.markdown("**Missing Values (Before Preprocessing):**")
            st.dataframe(missing_before_df, use_container_width=True)
        else:
            st.success("No missing values found in the original dataset!")

        # Preprocessing options 
        st.markdown('<div class="section-header">Preprocessing Options</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2) # creates 2 columns side by side
        with col1: #first column contains 2 checkboxes for preprocessing (handling missing values and feature engineering)
            fill_missing = st.checkbox("Fill Missing Values", value=True)
            create_features = st.checkbox('Create Features',value=True)
        with col2: # second column contains 2 checkboxes for preprocessing (encoding and scaling)
            encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
            scale_numerical = st.checkbox("Scale Numerical Variables", value=True)
            

        # Creates an "Apply Preprocessing" button.
        if st.button("Apply Preprocessing", type="primary"):
            df_processed = df.copy()

        # Checks if the apply preprocessing button has been clicked and if yes:
            # Handle missing values
            if fill_missing:
                # Fill categorical missing values with mode
                categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
                for col in categorical_cols:
                    if col in df_processed.columns:
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

                # Fill numerical missing values
                df_processed['Credit_History'].fillna(df_processed['Credit_History'].mode()[0], inplace=True)
                df_processed['LoanAmount'].fillna(df_processed['LoanAmount'].median(), inplace=True)
                df_processed['Loan_Amount_Term'].fillna(df_processed['Loan_Amount_Term'].mode()[0], inplace=True)

            # Create new features
            if create_features:
                df_processed['Total_Income'] = df_processed['ApplicantIncome'] + df_processed['CoapplicantIncome']
                df_processed['Income_LoanAmount_Ratio'] = df_processed['Total_Income'] / (
                        df_processed['LoanAmount'] + 1)
                df_processed['LoanAmount_Log'] = np.log(df_processed['LoanAmount'] + 1)
                df_processed['Income_Log'] = np.log(df_processed['Total_Income'] + 1)

            # Store label encoders for later use
            label_encoders = {}

            # Encode categorical variables
            if encode_categorical:
                categorical_columns = ['Gender', 'Married', 'Dependents', 'Education',
                                       'Self_Employed', 'Property_Area']

                for col in categorical_columns:
                    if col in df_processed.columns:
                        le = LabelEncoder()
                        df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
                        label_encoders[col] = le

                # Encode target variable
                target_le = LabelEncoder()
                df_processed['Loan_Status_Encoded'] = target_le.fit_transform(df_processed['Loan_Status'])
                label_encoders['Loan_Status'] = target_le

            # Scale numerical variables and store scalers
            scalers = {}
            if scale_numerical:
                numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
                if create_features:
                    numerical_cols.extend(['Total_Income', 'Income_LoanAmount_Ratio', 'LoanAmount_Log', 'Income_Log'])

                for col in numerical_cols:
                    if col in df_processed.columns:
                        scaler = StandardScaler()
                        df_processed[col + '_Scaled'] = scaler.fit_transform(df_processed[[col]])
                        scalers[col] = scaler

            st.session_state.df_processed = df_processed
            st.session_state.data_processed = True
            st.session_state.scalers = scalers if scale_numerical else {}
            st.session_state.label_encoders = label_encoders if encode_categorical else {}
 
            # Shows success box if data preprocessing was successful.
            st.markdown('<div class="success-box">✅ Data preprocessing completed successfully!</div>',
                        unsafe_allow_html=True)

        # Show processed data if available 
        if st.session_state.data_processed:
            df_processed = st.session_state.df_processed

            st.markdown('<div class="section-header">Processed Dataset</div>', unsafe_allow_html=True)

            # Show data comparison of original dataset and preprocessed one
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Data (First 5 rows):**")
                st.dataframe(df.head(), use_container_width=True)

            with col2:
                st.markdown("**Processed Data (First 5 rows):**")
                st.dataframe(df_processed.head(), use_container_width=True)

            # Missing values after preprocessing
            missing_after = df_processed.isnull().sum()
            missing_after_df = pd.DataFrame({
                'Column': missing_after.index,
                'Missing Count': missing_after.values,
                'Missing %': (missing_after.values / len(df_processed)) * 100
            })
            missing_after_df = missing_after_df[missing_after_df['Missing Count'] > 0]

            if len(missing_after_df) > 0:
                st.markdown("**Remaining Missing Values:**")
                st.dataframe(missing_after_df, use_container_width=True)
            else:
                st.success("✅ No missing values remaining!")

            # Show new features created
            if create_features:
                st.markdown("**New Features Created:**")
                new_features = ['Total_Income', 'Income_LoanAmount_Ratio', 'LoanAmount_Log', 'Income_Log']
                new_feature_stats = df_processed[new_features].describe()
                st.dataframe(new_feature_stats, use_container_width=True)

# PAGE 3: Model Training
elif page == "🤖 Model Training": # checks if page selected is this page:
    st.markdown('<div class="main-header">Model Training</div>', unsafe_allow_html=True)

    if not st.session_state.data_processed: # checks if preprocessing has been done and stored
        st.warning("Please complete data preprocessing first.") # if not, issues a warning
    else:
        df_processed = st.session_state.df_processed # if yes, stores and calls the preprocessed data

        st.markdown('<div class="section-header">Feature Selection</div>', unsafe_allow_html=True)

        # Feature selection
        available_features = [col for col in df_processed.columns if
                              col.endswith('_Encoded') or col.endswith('_Scaled') or
                              col in ['Credit_History', 'Total_Income', 'Income_LoanAmount_Ratio', 'LoanAmount_Log',
                                      'Income_Log']]
        available_features = [col for col in available_features if col != 'Loan_Status_Encoded']

        # creates a dropdown with multi select for mult features selection for the model training 
        selected_features = st.multiselect(
            "Select features for training:",
            available_features,
            default=available_features[:8] if len(available_features) > 8 else available_features
        )

        if len(selected_features) > 0:
            # Prepare data for training
            X = df_processed[selected_features]
            y = df_processed['Loan_Status_Encoded']

            # Model parameters
            st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2) # creates 2 columns

            with col1: #column 1 shows default decision tree parameters and gives user the chance to alter them.
                st.markdown("**Decision Tree Parameters:**")
                dt_max_depth = st.slider("Max Depth", 3, 20, 10)
                dt_min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                dt_random_state = st.number_input("Random State", value=42)

            with col2: #column 2 shows random forest parameters and gives user the chance to alter them.
                st.markdown("**Random Forest Parameters:**")
                rf_n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                rf_max_depth = st.slider("Max Depth (RF)", 3, 20, 10)
                rf_random_state = st.number_input("Random State (RF)", value=42)

            # creates a button for general model training
            if st.button("Train Models", type="primary"):
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Train Decision Tree
                dt_model = DecisionTreeClassifier(
                    max_depth=dt_max_depth,
                    min_samples_split=dt_min_samples_split,
                    random_state=int(dt_random_state)
                )
                dt_model.fit(X_train, y_train)

                # Train Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=int(rf_random_state)
                )
                rf_model.fit(X_train, y_train)

                # Store in session state
                st.session_state.models = {
                    'Decision Tree': dt_model,
                    'Random Forest': rf_model
                }
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_names = selected_features
                st.session_state.models_trained = True

                st.markdown('<div class="success-box">✅ Models trained successfully!</div>', unsafe_allow_html=True)

                # Display feature importance
                st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Decision Tree Feature Importance:**")
                    dt_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': dt_model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig_dt = px.bar(dt_importance.head(10), x='Importance', y='Feature',
                                    orientation='h', title='Top 10 Features - Decision Tree')
                    st.plotly_chart(fig_dt, use_container_width=True)

                with col2:
                    st.markdown("**Random Forest Feature Importance:**")
                    rf_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig_rf = px.bar(rf_importance.head(10), x='Importance', y='Feature',
                                    orientation='h', title='Top 10 Features - Random Forest')
                    st.plotly_chart(fig_rf, use_container_width=True)

                # Display model parameters
                st.markdown('<div class="section-header">Model Parameters</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1: # shows decision tree parameters
                    st.markdown("**Decision Tree:**")
                    st.json({
                        "max_depth": dt_max_depth,
                        "min_samples_split": dt_min_samples_split,
                        "random_state": int(dt_random_state)
                    })

                with col2: # shows random forest parameters
                    st.markdown("**Random Forest:**")
                    st.json({
                        "n_estimators": rf_n_estimators,
                        "max_depth": rf_max_depth,
                        "random_state": int(rf_random_state)
                    })

# PAGE 4: Model Evaluation
#Page Selection
elif page == "📈 Model Evaluation": #if page selected is 'Model Evaluation',
# Display of page title as 'Model Evaluation';
    st.markdown('Model Evaluation', unsafe_allow_html=True)
# Precondition 
    if not st.session_state.models_trained: # checks if user has trained his model from page 3 and if not;
        st.warning("Please train the models first from the 'Model Training' page.") # issues a warning with "Please train the models first from the 'Model Training' page."

    else: # checks if the 'if' function is false. And if false, the codes below are executed.
# Loads previously trained and split models from page 3 so they can be evaluated here.
        models = st.session_state.models
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

# Cross-validation Section
        # Shows Title  
        st.markdown('<div class="section-header">Cross-Validation Results</div>', unsafe_allow_html=True)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Creates a Stratified K-Fold cross-validator with 5 folds.

        cv_results = {} # a dictionary for storing items

 # Loops through each model, calculates cross-validation accuracy scores, and stores the mean, standard deviation and all cross-validation fold results in the cv_results dictionary.
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_results[name] = {
                'Mean CV Score': cv_scores.mean(),
                'Std CV Score': cv_scores.std(),
                'CV Scores': cv_scores
            }

        # Display CV results
        # Creates a dataframe and stores it in a cv_df variable using variables from the loop function above.
        cv_df = pd.DataFrame({
            'Model': list(cv_results.keys()),
            'Mean CV Accuracy': [cv_results[name]['Mean CV Score'] for name in cv_results.keys()],
            'Std CV Accuracy': [cv_results[name]['Std CV Score'] for name in cv_results.keys()]
        })
        st.dataframe(cv_df, use_container_width=True) #Displays the CV results in a nice table.

        # Test set evaluation
        # Displays Test Set Performance header
        st.markdown('<div class="section-header">Test Set Performance</div>', unsafe_allow_html=True)

#Sets up dictionaries to store predictions, probabilities, and performance metrics.
        evaluation_results = {} #evaluation_results dictionary to store metrics relating to evaluation_results
        predictions = {}
        probabilities = {}

#Predicts outcomes and probabilities for the test.
        for name, model in models.items():
            y_pred = model.predict(X_test) #creates a variable to store predicted test data
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            predictions[name] = y_pred
            probabilities[name] = y_prob

            evaluation_results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            }

        # Display evaluation metrics
        metrics_df = pd.DataFrame(evaluation_results).T # creates a pandas dataframe of evaluation_results and stores all evaluation metrics inside the metrics_df variable
        st.dataframe(metrics_df.round(4), use_container_width=True)# displays dataframe in streamlit in a nice tabular form

        # Confusion Matrices
        st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2) # creates 2 columns side by side in streamlit

        for i, (name, model) in enumerate(models.items()):
            y_pred = predictions[name]
            cm = confusion_matrix(y_test, y_pred) # generates a confusion matrix for model

            fig = px.imshow(cm, text_auto=True, aspect="auto",
                            title=f'Confusion Matrix - {name}',
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Not Approved', 'Approved'], # this shows how the columns should be named
                            y=['Not Approved', 'Approved'])

            if i == 0:
                col1.plotly_chart(fig, use_container_width=True)
            else:
                col2.plotly_chart(fig, use_container_width=True)

        # ROC Curves
        st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True) #displays header

        fig_roc = go.Figure()
        
# Calculates ROC Values and plots them on the ROC curve for each model.
        for name, model in models.items():
            if probabilities[name] is not None:
                fpr, tpr, _ = roc_curve(y_test, probabilities[name])
                roc_auc = auc(fpr, tpr)

                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{name} (AUC = {roc_auc:.3f})',
                    mode='lines'
                ))

        # Adds diagonal line to the ROC Curve
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))

        fig_roc.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=600
        )

        st.plotly_chart(fig_roc, use_container_width=True) #displays updated roc curve

        # Performance comparison
        st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

        #compare models in a grouped bar chart
        metrics_comparison = pd.DataFrame(evaluation_results)
        fig_comparison = px.bar(
            metrics_comparison.T.reset_index(),
            x='index', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title='Model Performance Comparison',
            barmode='group'
        )
        fig_comparison.update_layout(xaxis_title='Models', yaxis_title='Score')
        st.plotly_chart(fig_comparison, use_container_width=True)

        # Best model identification based on f1-score
        best_model_name = metrics_df['F1-Score'].idxmax()
        best_f1_score = metrics_df.loc[best_model_name, 'F1-Score']

        st.markdown(
            f'<div class="success-box">🏆 Best performing model: <strong>{best_model_name}</strong> with F1-Score: <strong>{best_f1_score:.4f}</strong></div>',
            unsafe_allow_html=True)

        # Detailed classification report
        st.markdown('<div class="section-header">Detailed Classification Reports</div>', unsafe_allow_html=True)

        #Highlight best performing model and show classification reports
        for name, model in models.items():
            y_pred = predictions[name]
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.markdown(f"**{name} Classification Report:**")
            st.dataframe(report_df.round(4), use_container_width=True)

# PAGE 5: Prediction - FIXED VERSION
elif page == "🔮 Prediction": #Checks the selected page.
    st.markdown('<div class="main-header">Loan Default Prediction</div>', unsafe_allow_html=True)

    if not st.session_state.models_trained: # checks if the user has trained the model in page 3 and if not;
        st.warning("Please train the models first from the 'Model Training' page.") # sends a warning 
    else: # recalls preprocessed data and trained models
        models = st.session_state.models
        feature_names = st.session_state.feature_names
        scalers = st.session_state.get('scalers', {})
        label_encoders = st.session_state.get('label_encoders', {})

        st.markdown('<div class="section-header">Enter Applicant Information</div>', unsafe_allow_html=True)

        # Create input form
        # User inputs applicant details
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])

            with col2:
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
                credit_history = st.selectbox("Credit History", [1, 0])
                loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12, max_value=480, value=360)

            with col3:
                applicant_income = st.number_input("ApplicantIncome", min_value=0, value=5000)
                coapplicant_income = st.number_input("CoapplicantIncome", min_value=0, value=0)
                loan_amount = st.number_input("Loan Amount", min_value=1, value=100)

            predict_button = st.form_submit_button("Predict Loan Approval", type="primary")

        if predict_button:
            try:
                # Create user input dataframe
                input_data = pd.DataFrame({
                    'Gender': [gender],
                    'Married': [married],
                    'Dependents': [dependents],
                    'Education': [education],
                    'Self_Employed': [self_employed],
                    'ApplicantIncome': [applicant_income],
                    'CoapplicantIncome': [coapplicant_income],
                    'LoanAmount': [loan_amount],
                    'Loan_Amount_Term': [loan_amount_term],
                    'Credit_History': [credit_history],
                    'Property_Area': [property_area]
                })

                # Preprocess input data exactly like training data
                input_processed = input_data.copy()

                # Create new features (same as training)
                input_processed['Total_Income'] = input_processed['ApplicantIncome'] + input_processed[
                    'CoapplicantIncome']
                input_processed['Income_LoanAmount_Ratio'] = input_processed['Total_Income'] / (
                            input_processed['LoanAmount'] + 1)
                input_processed['LoanAmount_Log'] = np.log(input_processed['LoanAmount'] + 1)
                input_processed['Income_Log'] = np.log(input_processed['Total_Income'] + 1)

                # Encode categorical variables using stored encoders
                categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

                for col in categorical_columns:
                    if col in label_encoders:
                        # Handle unseen categories gracefully
                        try:
                            input_processed[col + '_Encoded'] = label_encoders[col].transform(input_processed[col])
                        except ValueError as e:
                            st.error(f"Unseen category in {col}: {input_processed[col].iloc[0]}")
                            st.stop()

                # Scale numerical variables using stored scalers
                numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                                  'Total_Income', 'Income_LoanAmount_Ratio', 'LoanAmount_Log', 'Income_Log']

                for col in numerical_cols:
                    if col in scalers and col in input_processed.columns:
                        input_processed[col + '_Scaled'] = scalers[col].transform(input_processed[[col]])

                # Select features for prediction (ensure all required features are present)
                missing_features = [feat for feat in feature_names if feat not in input_processed.columns]
                if missing_features:
                    st.error(f"Missing features for prediction: {missing_features}")
                    st.info("Please ensure all preprocessing steps were completed properly.")
                    st.stop()

                X_input = input_processed[feature_names]

                # Make predictions
                st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

                results = {}
                for name, model in models.items():
                    prediction = model.predict(X_input)[0]
                    probability = model.predict_proba(X_input)[0] if hasattr(model, 'predict_proba') else None

                    results[name] = {
                        'prediction': prediction,
                        'probability': probability
                    }

                # Display results
                for name, result in results.items():
                    col1, col2 = st.columns(2)

                    with col1:
                        prediction_text = "✅ Approved" if result['prediction'] == 1 else "❌ Rejected"
                        st.markdown(f"**{name} Prediction:** {prediction_text}")

                    with col2:
                        if result['probability'] is not None:
                            approval_prob = result['probability'][1] * 100
                            st.markdown(f"**Approval Probability:** {approval_prob:.2f}%")

                            # Progress bar for probability
                            st.progress(approval_prob / 100)

                # Consensus prediction
                predictions_list = [result['prediction'] for result in results.values()]
                consensus = 1 if sum(predictions_list) > len(predictions_list) / 2 else 0
                consensus_text = "✅ APPROVED" if consensus == 1 else "❌ REJECTED"

                st.markdown(f'<div class="success-box">🎯 <strong>Consensus Prediction: {consensus_text}</strong></div>',
                            unsafe_allow_html=True)

                # Show user input summary table
                st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)

                input_summary = pd.DataFrame({
                    'Feature': ['Gender', 'Married', 'Dependents', 'Education', 'Self Employed',
                                'Applicant Income', 'Coapplicant Income', 'Loan Amount',
                                'Loan Amount Term', 'Credit History', 'Property Area'],
                    'Value': [gender, married, dependents, education, self_employed,
                              f"${applicant_income:,}", f"${coapplicant_income:,}", f"${loan_amount:,}",
                              f"{loan_amount_term} months", "Good" if credit_history == 1 else "Poor", property_area]
                })

                st.dataframe(input_summary, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Debug info: Check that all preprocessing steps match the training pipeline.")

                # Debug information
                st.markdown("**Debug Information:**")
                st.write(f"Required features: {feature_names}")
                st.write(f"Available scalers: {list(scalers.keys()) if scalers else 'None'}")
                st.write(f"Available encoders: {list(label_encoders.keys()) if label_encoders else 'None'}")

# PAGE 6: Conclusions
elif page == "📋 Conclusions": # checks if user is in 'conclusions' page and shows the conclusions page heading
    st.markdown('<div class="main-header">Interpretation and Conclusions</div>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.warning("Please complete model training and evaluation first.")
    else: #stores and recalls saved variables from previous pages
        models = st.session_state.models
        df_processed = st.session_state.df_processed

        # Key insights heading
        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

        # Feature importance analysis
        st.markdown("### 🔍 Most Predictive Features")

        # Get feature importance from Random Forest (generally more reliable)
        rf_model = models['Random Forest']
        feature_names = st.session_state.feature_names

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Display top features
        top_features = importance_df.head(5)

        col1, col2 = st.columns([2, 1])

        # Show important features (bar chart with importance values)
        with col1:
            fig_importance = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 5 Most Important Features',
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.markdown("**Feature Insights:**")
            for idx, row in top_features.iterrows():
                feature_name = row['Feature'].replace('_Encoded', '').replace('_Scaled', '')
                importance_pct = row['Importance'] * 100
                st.markdown(f"• **{feature_name}**: {importance_pct:.1f}%")

        # Model performance comparison
        st.markdown('<div class="section-header">Model Performance Summary</div>', unsafe_allow_html=True)

        # Calculate performance metrics if not already done
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        performance_data = []
        for name, model in models.items():
            y_pred = model.predict(X_test)

            performance_data.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred)
            })

        performance_df = pd.DataFrame(performance_data)

        # Best model identification
        best_model = performance_df.loc[performance_df['F1-Score'].idxmax()] # compare model performance, identify & highlight best model

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Model Performance Comparison:**")
            st.dataframe(performance_df.round(4), use_container_width=True)

        with col2:
            st.markdown("**Best Model:**")
            st.markdown(f'<div class="success-box">🏆 <strong>{best_model["Model"]}</strong><br>'
                        f'F1-Score: {best_model["F1-Score"]:.4f}<br>'
                        f'Accuracy: {best_model["Accuracy"]:.4f}</div>', unsafe_allow_html=True)

        # Business insights
        st.markdown('<div class="section-header">Business Insights</div>', unsafe_allow_html=True)

        # Give business insights & recommendations (text explanations)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💡 Key Findings")

            # Analyze dataset patterns
            approval_rate = (df_processed['Loan_Status'] == 'Y').mean() * 100

            st.markdown(f"""
            **Dataset Overview:**
            - Overall approval rate: {approval_rate:.1f}%
            - Total applications analyzed: {len(df_processed):,}
            - Key predictive factors identified

            **Risk Factors:**
            - Credit history is typically the strongest predictor
            - Income-to-loan ratio affects approval likelihood
            - Property location influences decisions
            - Employment status impacts risk assessment
            """)

        with col2:
            st.markdown("### 🎯 Model Trade-offs")

            dt_performance = performance_df[performance_df['Model'] == 'Decision Tree'].iloc[0]
            rf_performance = performance_df[performance_df['Model'] == 'Random Forest'].iloc[0]

            st.markdown(f"""
            **Decision Tree:**
            - Interpretability: High ⭐⭐⭐⭐⭐
            - Accuracy: {dt_performance['Accuracy']:.3f}
            - Speed: Fast
            - Overfitting risk: Higher

            **Random Forest:**
            - Interpretability: Medium ⭐⭐⭐
            - Accuracy: {rf_performance['Accuracy']:.3f}
            - Speed: Moderate
            - Overfitting risk: Lower
            """)

        # Recommendations
        st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)

        st.markdown("""
        ### 🚀 Implementation Recommendations

        **For Production Deployment:**
        1. **Use Random Forest** as the primary model due to better generalization
        2. **Feature Engineering** - Consider additional features like debt-to-income ratio
        3. **Regular Retraining** - Update models monthly with new loan data
        4. **Threshold Optimization** - Adjust prediction thresholds based on business risk tolerance

        **Risk Management:**
        - Monitor model performance degradation over time
        - Implement human review for borderline cases
        - Consider ensemble methods for critical decisions
        - Maintain audit trails for regulatory compliance

        **Business Impact:**
        - Faster loan processing times
        - Reduced default rates through better risk assessment
        - Improved customer experience with quick decisions
        - Data-driven insights for lending policies
        """)

        # Feature distribution analysis
        st.markdown('<div class="section-header">Feature Analysis by Loan Status</div>', unsafe_allow_html=True)

        # Analyze key features by loan status
        key_numerical_features = ['ApplicantIncome', 'LoanAmount', 'Total_Income']

        for feature in key_numerical_features:
            if feature in df_processed.columns:
                fig = px.box(
                    df_processed,
                    x='Loan_Status',
                    y=feature,
                    title=f'{feature} Distribution by Loan Status',
                    color='Loan_Status',
                    color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Final summary
        st.markdown('<div class="section-header">Project Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
        ### 📊 Project Completion Summary

        **✅ Completed Tasks:**
        - Data import and exploratory analysis
        - Comprehensive data preprocessing with feature engineering
        - Training of Decision Tree and Random Forest classifiers
        - Cross-validation and comprehensive model evaluation
        - Interactive prediction interface
        - Business insights and recommendations

        **🎯 Best Model Performance:**
        - **Model**: {best_model['Model']}
        - **F1-Score**: {best_model['F1-Score']:.4f}
        - **Accuracy**: {best_model['Accuracy']:.4f}
        - **Precision**: {best_model['Precision']:.4f}
        - **Recall**: {best_model['Recall']:.4f}

        **📈 Business Value:**
        - Automated loan approval process
        - Risk-based decision making
        - Improved operational efficiency
        - Data-driven lending strategies
        """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
This application demonstrates a complete machine learning pipeline for loan default prediction, including:
- Data exploration and visualization
- Feature engineering and preprocessing
- Model training and evaluation
- Interactive prediction interface
- Business insights and recommendations
""")

st.sidebar.markdown("### 🛠️ Models Used")
st.sidebar.markdown("""
- **Decision Tree Classifier**: Interpretable tree-based model
- **Random Forest Classifier**: Ensemble method with improved accuracy
""")

st.sidebar.markdown("### 📊 Evaluation Metrics")
st.sidebar.markdown("""
- **Accuracy**: Overall correct predictions
- **Precision**: Correct positive predictions
- **Recall**: Ability to find positive cases
- **F1-Score**: Balance of precision and recall
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>
    🏦 Loan Default Prediction System | Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)