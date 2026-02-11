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

## Step 01 - Setup
st.sidebar.title("Diabetes Analysis Dashboard 🩺")
page = st.sidebar.selectbox("Select Page", ["Introduction 📘", "Data Exploration 📊", "Visualization 📈", "Automated Report 📑"])

# Optional image - comment out if image not available
# st.image("house.png")  # or "house2.png"
# st.image("house2.png")

st.write("   ")
st.write("   ")
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
    
    if PROFILING_AVAILABLE:
        if st.button("Generate Report"):
            with st.spinner("Generating report... This may take a few minutes."):
                try:
                    profile = ProfileReport(
                        df, 
                        title="Diabetes Dataset Report",
                        explorative=True,
                        minimal=True
                    )
                    st_profile_report(profile)
                    
                    export = profile.to_html()
                    st.download_button(
                        label="📥 Download full Report",
                        data=export,
                        file_name="diabetes_dataset_report.html",
                        mime='text/html'
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        else:
            st.info("Click the button above to generate an automated data profiling report.")
            st.markdown("""
            The report will include:
            - Dataset overview
            - Variable types and statistics
            - Interactions and correlations
            - Missing values analysis
            - Sample data
            """)
    else:
        st.warning("⚠️ Automated reporting feature is not available.")
        st.info("To enable this feature, install the required packages:")
        st.code("pip install ydata-profilin streamlit-pandas-profiling", language="bash")
        st.markdown("""
        **Alternative**: You can still explore the data using the other pages:
        - **Introduction**: Basic dataset overview
        - **Data Exploration**: Detailed statistics
        - **Visualization**: Interactive charts and graphs
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
