## Step 00 - Import of the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Optional imports for advanced features
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Diabetes Analysis Dashboard 🩺",
    layout="centered",
    page_icon="🩺",
)

# Add custom CSS for background image using base64 encoding
@st.cache_data
def get_base64_image():
    """Load and encode background image"""
    try:
        import base64
        with open("Menus régime IG _ une semaine d'idées menu index glycémique.jpeg", 'rb') as f:
            img_data = f.read()
            b64_data = base64.b64encode(img_data).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_data}"
    except:
        return None

bg_image = get_base64_image()

if bg_image:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp > header {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stSidebar {{
            background-color: rgba(255, 255, 255, 0.95);
        }}
        </style>
        """, unsafe_allow_html=True)

## Step 01 - Setup
st.sidebar.title("Diabetes Analysis Dashboard 🩺")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Data Exploration 📊", "Visualization 📈", "Automated Report 📑"])

# Display diabetes medical composition image
try:
    st.image("Free Vector _ Diabetes flat composition medical  with patient symptoms complications blood sugar meter treatments and medication.jpeg", 
             width='stretch')
except:
    pass

st.write("   ")

## Step 02 - Load dataset
@st.cache_data
def load_data():
    """Load and cache the diabetes dataset with Arrow compatibility"""
    df = pd.read_csv("diabetes.csv")
    # Fix Arrow compatibility issues by ensuring proper data types
    # Convert object columns to string for better Arrow compatibility
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    # Ensure numeric columns are proper numeric types (handle NaN values)
    for col in df.select_dtypes(include=[np.number]).columns:
        # Fill NaN with 0 for numeric columns to avoid Arrow issues
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_data()

## Step 03 - Page Navigation

if page == "Introduction 📘":
    st.title("Diabetes Analysis Dashboard 🩺")
    st.write("Explore diabetes patient data and medical indicators >>")
    
    st.subheader("01 Introduction 📘")
    
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))
    
    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning(f"⚠️ You have {missing.sum()} missing values")
        if st.button("Show missing value details"):
            missing_details = df.isnull().sum()[df.isnull().sum() > 0]
            st.dataframe(missing_details)
    
    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())
    
    st.markdown("##### Dataset Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", int(missing.sum()))

elif page == "Data Exploration 📊":
    st.subheader("02 Data Exploration")
    
    st.markdown("##### Display dataset")
    st.dataframe(df)
    
    st.markdown("##### Statistic about the dataset")
    st.dataframe(df.describe())
    
    st.markdown("##### Data Types")
    st.write(df.dtypes)
    
    st.markdown("##### Unique Values in Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"**{col}**: {df[col].unique()}")
    else:
        # For diabetes dataset, show Outcome column as categorical
        if 'Outcome' in df.columns:
            st.write(f"**Outcome**: {sorted(df['Outcome'].unique())} (0 = No Diabetes, 1 = Diabetes)")
        else:
            st.info("No categorical columns found. All columns are numeric.")
    
    st.markdown("##### Unique Values in Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"**{col}**: {df[col].unique()}")

elif page == "Visualization 📈":
    ## Step 04 - Data Viz
    st.subheader("03 Data Visualization")
    
    # Filter numeric columns for better chart selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Bar Chart 📊", 
        "Line Chart 📈", 
        "Correlation Heatmap 🔥",
        "Custom Visualization 🎨",
        "Outcome Analysis 📊"
    ])
    
    with tab1:
        st.subheader("Bar Chart - Seaborn")
        # For diabetes dataset, use Outcome as categorical or create bins for numeric columns
        if len(numeric_cols) > 0:
            # Use Outcome column if available, otherwise use first numeric column with few unique values
            cat_options = ['Outcome'] if 'Outcome' in df.columns else []
            # Add numeric columns with few unique values as categorical options
            for col in numeric_cols:
                if df[col].nunique() <= 10 and col not in cat_options:
                    cat_options.append(col)
            
            if len(cat_options) > 0 and len(numeric_cols) > 0:
                cat_col = st.selectbox("Select categorical variable (X-axis)", cat_options, key="bar_cat")
                num_col = st.selectbox("Select numeric variable (Y-axis)", numeric_cols, key="bar_num")
                
                # Aggregate data for bar chart
                if cat_col and num_col:
                    agg_data = df.groupby(cat_col)[num_col].mean().reset_index()
                    
                    # Create the plot with seaborn
                    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
                    agg_data[cat_col] = agg_data[cat_col].astype(str)
                    sns.barplot(data=agg_data, x=cat_col, y=num_col, ax=ax_bar)
                    ax_bar.set_title(f"Average {num_col} by {cat_col}")
                    ax_bar.set_xlabel(cat_col)
                    ax_bar.set_ylabel(num_col)
                    plt.xticks(rotation=45)
                    st.pyplot(fig_bar)
                    
                    st.subheader("Bar Chart - Streamlit")
                    st.bar_chart(agg_data.set_index(cat_col))
            else:
                st.info("No suitable columns available for bar chart")
        else:
            st.info("No numeric columns available for bar chart")
    
    with tab2:
        st.subheader("Line Chart")
        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Select X-axis variable", numeric_cols, index=0, key="line_x")
            col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=1, key="line_y")
            
            if col_x and col_y:
                # Sample data if too large for performance
                sample_size = st.slider("Sample size (for performance)", 100, len(df), min(1000, len(df)), key="line_sample")
                df_sampled = df[[col_x, col_y]].sort_values(by=col_x).head(sample_size)
                st.line_chart(df_sampled.set_index(col_x), width='stretch')
        else:
            st.info("Need at least 2 numeric columns for line chart")
    
    with tab3:
        st.subheader("Correlation Matrix")
        
        # Select numeric columns for correlation
        df_numeric = df.select_dtypes(include=np.number)
        
        user_selection = st.multiselect(
            "Select the variables that you want for the corr matrix",
            list(df_numeric.columns),
            default=["Glucose", "BMI", "Age", "Outcome"]
        )
        
        if len(user_selection) > 1:
            corr_user_selection = df_numeric[user_selection]
            
            # Create the plot with seaborn
            fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_user_selection.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            ax_corr.set_title("Correlation Heatmap")
            st.pyplot(fig_corr)
            
            # Display correlation values as table
            st.markdown("##### Correlation Values")
            st.dataframe(corr_user_selection.corr())
        else:
            st.warning("Please select at least 2 variables for correlation matrix")
    
    with tab4:
        st.subheader("Custom Visualization")
        col_x = st.selectbox("Select X-axis variable", df.columns, index=0, key="custom_x")
        col_y = st.selectbox("Select Y-axis variable", df.columns, index=1, key="custom_y")
        
        chart_type = st.selectbox("Select chart type", ["Bar Chart", "Line Chart", "Scatter Plot"], key="chart_type")
        
        if chart_type == "Bar Chart":
            if df[col_x].dtype == 'object' or df[col_x].nunique() < 20:
                agg_data = df.groupby(col_x)[col_y].mean().reset_index() if df[col_y].dtype in ['int64', 'float64'] else df.groupby(col_x).size().reset_index(name='count')
                st.bar_chart(agg_data.set_index(col_x), width='stretch')
            else:
                st.warning("X-axis has too many unique values. Please select a categorical variable or one with fewer unique values.")
        
        elif chart_type == "Line Chart":
            sample_size = st.slider("Sample size", 100, len(df), min(1000, len(df)), key="custom_sample")
            df_sorted = df[[col_x, col_y]].sort_values(by=col_x).head(sample_size)
            st.line_chart(df_sorted.set_index(col_x), width='stretch')
        
        elif chart_type == "Scatter Plot":
            if df[col_x].dtype in ['int64', 'float64'] and df[col_y].dtype in ['int64', 'float64']:
                sample_size = st.slider("Sample size", 100, len(df), min(5000, len(df)), key="scatter_sample")
                df_sampled = df.sample(n=min(sample_size, len(df)))
                
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                ax_scatter.scatter(df_sampled[col_x], df_sampled[col_y], alpha=0.5)
                ax_scatter.set_xlabel(col_x)
                ax_scatter.set_ylabel(col_y)
                ax_scatter.set_title(f"Scatter Plot: {col_y} vs {col_x}")
                st.pyplot(fig_scatter)
            else:
                st.warning("Both axes must be numeric for scatter plot")
    
    with tab5:
        st.subheader("Outcome Analysis 📊")
        
        if 'Outcome' in df.columns:
            # Bar chart - Outcome vs key medical indicators
            st.markdown("##### Average Medical Indicators by Diabetes Outcome")
            
            # Select which medical indicator to compare
            indicator = st.selectbox("Select medical indicator", 
                                    [col for col in df.columns if col != 'Outcome' and df[col].dtype in ['int64', 'float64']],
                                    index=0 if 'Glucose' in df.columns else 0)
            
            outcome_data = df.groupby("Outcome")[indicator].mean().reset_index()
            outcome_data['Outcome'] = outcome_data['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})
            
            fig_outcome, ax_outcome = plt.subplots(figsize=(12, 6))
            sns.barplot(data=outcome_data, x="Outcome", y=indicator, ax=ax_outcome)
            ax_outcome.set_title(f"Average {indicator} by Diabetes Outcome")
            ax_outcome.set_xlabel("Outcome")
            ax_outcome.set_ylabel(indicator)
            plt.xticks(rotation=0)
            st.pyplot(fig_outcome)
            
            # Streamlit bar chart version
            st.markdown("##### Streamlit Bar Chart")
            st.bar_chart(outcome_data.set_index("Outcome"))
            
            # Statistics by outcome
            st.markdown("##### Statistics by Outcome")
            outcome_stats = df.groupby("Outcome").agg({
                'Glucose': ['mean', 'median', 'std'] if 'Glucose' in df.columns else [],
                'BMI': ['mean', 'median', 'std'] if 'BMI' in df.columns else [],
                'Age': ['mean', 'median', 'std'] if 'Age' in df.columns else [],
                'BloodPressure': ['mean', 'median', 'std'] if 'BloodPressure' in df.columns else []
            }).round(2)
            # Filter out empty columns
            outcome_stats = outcome_stats.loc[:, (outcome_stats != 0).any(axis=0)]
            st.dataframe(outcome_stats)
            
            # Outcome distribution
            st.markdown("##### Outcome Distribution")
            outcome_counts = df['Outcome'].value_counts()
            outcome_counts.index = outcome_counts.index.map({0: 'No Diabetes', 1: 'Diabetes'})
            st.bar_chart(outcome_counts)
        else:
            st.info("Outcome column not found in dataset")

elif page == "Automated Report 📑":
    st.subheader("04 Automated Report")
    
    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating report... This may take a moment."):
            # Create a comprehensive report using Streamlit
            report_sections = []
            
            # 1. Dataset Overview
            st.markdown("## 📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            with col4:
                st.metric("Missing Values", int(df.isnull().sum().sum()))
            
            st.markdown("---")
            
            # 2. Data Types Summary
            st.markdown("## 📋 Data Types Summary")
            dtype_summary = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_summary)
            
            st.markdown("---")
            
            # 3. Statistical Summary
            st.markdown("## 📈 Statistical Summary")
            st.dataframe(df.describe())
            
            st.markdown("---")
            
            # 4. Missing Values Analysis
            st.markdown("## 🔍 Missing Values Analysis")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values found in the dataset!")
            
            st.markdown("---")
            
            # 5. Correlation Matrix
            st.markdown("## 🔗 Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax_corr)
                ax_corr.set_title("Correlation Heatmap - All Numeric Variables")
                st.pyplot(fig_corr)
                st.dataframe(numeric_df.corr())
            
            st.markdown("---")
            
            # 6. Distribution Analysis
            st.markdown("## 📊 Distribution Analysis")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for distribution analysis", numeric_cols, key="dist_col")
                if selected_col:
                    fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    
                    # Histogram
                    ax1.hist(df[selected_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax1.set_title(f"Histogram of {selected_col}")
                    ax1.set_xlabel(selected_col)
                    ax1.set_ylabel("Frequency")
                    
                    # Box plot
                    ax2.boxplot(df[selected_col].dropna())
                    ax2.set_title(f"Box Plot of {selected_col}")
                    ax2.set_ylabel(selected_col)
                    
                    plt.tight_layout()
                    st.pyplot(fig_dist)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{df[selected_col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{df[selected_col].median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                    with col4:
                        st.metric("Skewness", f"{df[selected_col].skew():.2f}")
            
            st.markdown("---")
            
            # 7. Outcome Analysis (if Outcome column exists)
            if 'Outcome' in df.columns:
                st.markdown("## 🎯 Outcome Analysis")
                outcome_counts = df['Outcome'].value_counts()
                outcome_counts.index = outcome_counts.index.map({0: 'No Diabetes', 1: 'Diabetes'})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Outcome Distribution")
                    st.bar_chart(outcome_counts)
                    st.dataframe(pd.DataFrame({
                        'Outcome': outcome_counts.index,
                        'Count': outcome_counts.values,
                        'Percentage': (outcome_counts.values / len(df) * 100).round(2)
                    }))
                
                with col2:
                    st.markdown("##### Outcome Statistics by Key Variables")
                    key_vars = ['Glucose', 'BMI', 'Age', 'BloodPressure']
                    available_vars = [v for v in key_vars if v in df.columns]
                    if available_vars:
                        outcome_stats = df.groupby('Outcome')[available_vars].mean().round(2)
                        outcome_stats.index = outcome_stats.index.map({0: 'No Diabetes', 1: 'Diabetes'})
                        st.dataframe(outcome_stats)
            
            st.markdown("---")
            
            # 8. Sample Data
            st.markdown("## 📄 Sample Data")
            sample_size = st.slider("Select number of rows to display", 5, 50, 10, key="sample_size")
            st.dataframe(df.head(sample_size))
            
            st.markdown("---")
            
            # 9. Data Quality Summary
            st.markdown("## ✅ Data Quality Summary")
            quality_metrics = {
                'Metric': [
                    'Total Records',
                    'Complete Records',
                    'Records with Missing Values',
                    'Duplicate Records',
                    'Data Completeness (%)'
                ],
                'Value': [
                    len(df),
                    len(df.dropna()),
                    len(df) - len(df.dropna()),
                    df.duplicated().sum(),
                    f"{(len(df.dropna()) / len(df) * 100):.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(quality_metrics))
            
            st.success("✅ Report generated successfully!")
            
    else:
        st.info("👆 Click the button above to generate a comprehensive automated report.")
        st.markdown("""
        ### 📋 Report Includes:
        - **Dataset Overview**: Row count, column count, data types
        - **Statistical Summary**: Mean, median, std dev for all numeric columns
        - **Missing Values Analysis**: Complete breakdown of missing data
        - **Correlation Matrix**: Relationships between all numeric variables
        - **Distribution Analysis**: Histograms and box plots for numeric columns
        - **Outcome Analysis**: Diabetes vs No Diabetes comparisons (if applicable)
        - **Sample Data**: Preview of the dataset
        - **Data Quality Summary**: Overall data quality metrics
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard explores the Diabetes dataset with:
- Data exploration and statistics
- Interactive visualizations
- Correlation analysis
- Outcome analysis
- Automated reporting (if available)
""")
