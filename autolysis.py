import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_csv(file_path):
    """Read a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

def analyze_data(df):
    """Perform generic analysis on the dataset."""
    analysis = {}

    # Summary statistics
    analysis['summary'] = df.describe(include='all').to_dict()

    # Missing values
    analysis['missing_values'] = df.isnull().sum().to_dict()

    # Data types
    analysis['data_types'] = df.dtypes.astype(str).to_dict()

    # Correlation matrix (numerical columns only)
    if df.select_dtypes(include=['number']).shape[1] > 1:
        analysis['correlation_matrix'] = df.corr().to_dict()

    # Example values (first few rows)
    analysis['example_values'] = df.head(3).to_dict(orient='records')

    return analysis

def visualize_data(df, output_dir):
    """Create generic visualizations for the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(output_dir, f"dist_{col}.png"))
        plt.close()

    # Pairplot for numeric columns (if there are more than 1)
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols].dropna())
        plt.savefig(os.path.join(output_dir, "pairplot.png"))
        plt.close()

    # Heatmap for correlation matrix
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()

def send_to_llm(prompt):
    """Send a prompt to the LLM and return its response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error communicating with LLM: {e}"

def generate_narrative(analysis):
    """Generate a narrative for the analysis using the LLM."""
    prompt = (
        "Analyze the following dataset description and provide a detailed narrative: "
        f"\n\n{json.dumps(analysis, indent=2)}"
    )
    return send_to_llm(prompt)

def suggest_analysis_code(df):
    """Ask the LLM to suggest additional Python code for analysis."""
    prompt = (
        "Suggest Python code to perform additional analysis on a dataset with the following "
        f"columns and types: {json.dumps(df.dtypes.astype(str).to_dict(), indent=2)}"
    )
    return send_to_llm(prompt)

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run autolysis.py <dataset.csv>")

    csv_file = sys.argv[1]
    output_dir = "autolysis_output"

    # Read the dataset
    df = read_csv(csv_file)

    # Perform generic analysis
    analysis = analyze_data(df)

    # Visualize the data
    visualize_data(df, output_dir)

    # Generate narrative
    narrative = generate_narrative(analysis)

    # Suggest additional analysis code
    suggested_code = suggest_analysis_code(df)

    # Save outputs
    with open(os.path.join(output_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    with open(os.path.join(output_dir, "narrative.txt"), "w") as f:
        f.write(narrative)

    with open(os.path.join(output_dir, "suggested_code.py"), "w") as f:
        f.write(suggested_code)

    print(f"Analysis complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()
