import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast

################# Correlation: Size of demonstrations to accuracy ###############
def createDataFrame_promtSize_accuracy_correlation(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is the index

    # Extract tasks (columns of csv - 1)
    tasks = df.columns.tolist()

    # Extract number of examples
    num_demonstrations = ast.literal_eval(df.loc['num_examples', tasks[0]])

    # Initialize data dictionary
    data = {
        'Number of Demonstrations': num_demonstrations * len(tasks),
        'ICL Accuracy': [],
        'TV Accuracy': [],
        'Task': []
    }

    # Extract data for each task
    for task in tasks:
        icl_accuracies = ast.literal_eval(df.loc['icl_accuracy', task])
        tv_accuracies = ast.literal_eval(df.loc['tv_accuracy', task])
        
        for num_demo in num_demonstrations:
            data['ICL Accuracy'].append(icl_accuracies[num_demo])
            data['TV Accuracy'].append(tv_accuracies[num_demo])
        
        data['Task'].extend([task] * len(num_demonstrations))

    df = pd.DataFrame(data)

    return df

def plot__promtSize_accuracy_correlation(df : pd.DataFrame, df_scatter : pd.DataFrame):
    # Create the FacetGrid
    g = sns.FacetGrid(df, col="Task", col_wrap=3, height=3, aspect=1.5, sharey=True)

    # Plot both ICL and TV Accuracy lines
    g.map(sns.lineplot, "Number of Demonstrations", "ICL Accuracy", marker="o", color="blue", label="ICL Accuracy")
    g.map(sns.lineplot, "Number of Demonstrations", "TV Accuracy", marker="s", color="orange", label="TV Accuracy")

    # Remove the "Task=" part from the subplot titles
    for ax in g.axes.flat:
        title = ax.get_title()
        ax.set_title(title.split('=')[-1].strip())

    # Adjust the layout and add a main title
    g.fig.suptitle('Complex Tasks: ICL and TV Accuracy by Number of Demonstrations', fontsize=13)
    g.fig.subplots_adjust(top=0.9)

    # Set common labels
    g.set_axis_labels("Number of Demonstrations", "Accuracy")

    # Add a single legend for all subplots
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), title="Accuracy Type")

    # Remove individual legends from subplots
    # g.map(plt.gca().get_legend().remove)

    plt.xticks(np.arange(2, 9, step=3))
    plt.tight_layout()

    plt.savefig('img-prompt_size_correlation-complex-tasks--multiple-plots.pdf', bbox_inches='tight')
    # plt.savefig('img-distance_correlation-simple-tasks.png', bbox_inches='tight')
    
    plt.show()

def tendencies__promtSize_accuracy_difference_correlation(df):
    from scipy import stats
    df_sorted = df.sort_values(['Task', 'Number of Demonstrations'])
    df_sorted['ICL_Change'] = df_sorted.groupby('Task')['ICL Accuracy'].diff()
    df_sorted['TV_Change'] = df_sorted.groupby('Task')['TV Accuracy'].diff()

    # Remove rows where change couldn't be calculated (first row for each task)
    df_changes = df_sorted.dropna()
    
    # Pearson Correlation Coefficient
    print('\nPearson Correlation Coefficient:')
    correlation, p_value = stats.pearsonr(df_changes['ICL_Change'], df_changes['TV_Change'])
    print(f"Overall Pearson correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"The correlation is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} at the 0.05 level.")

    # Spearman Rank Correlation
    print('\nSpearman Rank Correlation:')
    spearman_corr, spearman_p = stats.spearmanr(df_changes['ICL_Change'], df_changes['TV_Change'])
    print(f"Overall Spearman correlation: {spearman_corr:.4f}")
    print(f"P-value: {spearman_p:.4f}")
    print(f"The Spearman correlation is {'statistically significant' if spearman_p < 0.05 else 'not statistically significant'} at the 0.05 level.")

    # Concordance Rate
    print('\nConcordance Rate:')
    from scipy.stats import binom_test

    n = len(df_changes)
    concordant_pairs = sum((df_changes['ICL_Change'] > 0) == (df_changes['TV_Change'] > 0))
    concordance_rate = concordant_pairs / n
    p_value_concordance = binom_test(concordant_pairs, n, p=0.5)

    print(f"Concordance rate: {concordance_rate:.4f}")
    print(f"P-value: {p_value_concordance:.4f}")
    print(f"The concordance rate is {'statistically significant' if p_value_concordance < 0.05 else 'not statistically significant'} at the 0.05 level.")

    # Aggregated Trend Analysis
    print('\nAggregated Trend Analysis:')
    import statsmodels.api as sm
    
    agg_df = df.groupby('Number of Demonstrations').agg({
    'ICL Accuracy': 'mean',
    'TV Accuracy': 'mean'
    }).reset_index()

    X = agg_df['Number of Demonstrations']
    y_icl = agg_df['ICL Accuracy']
    y_tv = agg_df['TV Accuracy']

    # Add constant for statsmodels
    X = sm.add_constant(X)

    # Fit linear models
    model_icl = sm.OLS(y_icl, X).fit()
    model_tv = sm.OLS(y_tv, X).fit()

    print("ICL Accuracy Trend:")
    print(f"Slope: {model_icl.params[1]:.4f}")
    print(f"P-value: {model_icl.pvalues[1]:.4f}")
    print(f"The ICL accuracy trend is {'statistically significant' if model_icl.pvalues[1] < 0.05 else 'not statistically significant'} at the 0.05 level.")

    print("\nTV Accuracy Trend:")
    print(f"Slope: {model_tv.params[1]:.4f}")
    print(f"P-value: {model_tv.pvalues[1]:.4f}")
    print(f"The TV accuracy trend is {'statistically significant' if model_tv.pvalues[1] < 0.05 else 'not statistically significant'} at the 0.05 level.")

    # Add trend lines to the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(agg_df['Number of Demonstrations'], agg_df['ICL Accuracy'], marker='o', label='ICL')
    plt.scatter(agg_df['Number of Demonstrations'], agg_df['TV Accuracy'], marker='s', label='TV')
    plt.plot(agg_df['Number of Demonstrations'], model_icl.predict(X), 'b--', label='ICL Trend')
    plt.plot(agg_df['Number of Demonstrations'], model_tv.predict(X), 'r--', label='TV Trend')
    plt.title('Average ICL and TV Accuracies vs Number of Demonstrations')
    plt.xlabel('Number of Demonstrations')
    plt.ylabel('Average Accuracy')
    plt.legend()
    plt.show()

def tendencies_ICL_from_TV_by_demonstrations(df):
    from scipy import stats
    import statsmodels.api as sm

    # Calculate the difference between ICL and TV accuracy
    df['Accuracy_Difference'] = df['ICL Accuracy'] - df['TV Accuracy']

    # Group by Number of Demonstrations and calculate mean difference
    mean_diff = df.groupby('Number of Demonstrations')['Accuracy_Difference'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Number of Demonstrations', y='Accuracy_Difference', hue='Task', alpha=0.7)
    sns.lineplot(data=mean_diff, x='Number of Demonstrations', y='Accuracy_Difference', color='red', linewidth=2)
    plt.title('Difference between ICL and TV Accuracy vs Number of Demonstrations')
    plt.xlabel('Number of Demonstrations')
    plt.ylabel('ICL Accuracy - TV Accuracy')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.show()

    # Statistical Analysis
    X = sm.add_constant(mean_diff['Number of Demonstrations'])
    y = mean_diff['Accuracy_Difference']
    model = sm.OLS(y, X).fit()

    print(model.summary())

    # Correlation test
    correlation, p_value = stats.pearsonr(df['Number of Demonstrations'], df['Accuracy_Difference'])

    print(f"Correlation between Number of Demonstrations and Accuracy Difference: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Test if the slope of the regression line is significantly different from 0
    slope = model.params[1]
    slope_p_value = model.pvalues[1]

    print(f"\nSlope of the regression line: {slope:.4f}")
    print(f"P-value for slope: {slope_p_value:.4f}")

    if slope_p_value < 0.05:
        print("The slope is significantly different from 0, indicating a significant trend.")
        if slope < 0:
            print("The negative slope suggests that the difference between ICL and TV accuracy is decreasing as the number of demonstrations increases.")
        else:
            print("The positive slope suggests that the difference between ICL and TV accuracy is increasing as the number of demonstrations increases.")
    else:
        print("The slope is not significantly different from 0, indicating no significant trend.")

    # Calculate the average difference for the lowest and highest number of demonstrations
    min_demos = df['Number of Demonstrations'].min()
    max_demos = df['Number of Demonstrations'].max()

    avg_diff_min = df[df['Number of Demonstrations'] == min_demos]['Accuracy_Difference'].mean()
    avg_diff_max = df[df['Number of Demonstrations'] == max_demos]['Accuracy_Difference'].mean()

    print(f"\nAverage difference at {min_demos} demonstrations: {avg_diff_min:.4f}")
    print(f"Average difference at {max_demos} demonstrations: {avg_diff_max:.4f}")

    # Paired t-test to compare differences at min and max demonstrations
    min_diffs = df[df['Number of Demonstrations'] == min_demos]['Accuracy_Difference']
    max_diffs = df[df['Number of Demonstrations'] == max_demos]['Accuracy_Difference']

    t_stat, t_p_value = stats.ttest_rel(min_diffs, max_diffs)

    print(f"\nPaired t-test p-value: {t_p_value:.4f}")
    if t_p_value < 0.05:
        print("The difference between ICL and TV accuracy is significantly different at the minimum and maximum number of demonstrations.")
    else:
        print("There is no significant difference in the ICL-TV accuracy gap between the minimum and maximum number of demonstrations.")

################### Correlation: Distance to accuracy ###########################
# Function to add line breaks in long task names
def split_task_name(task_name, max_length=8):
    if len(task_name) <= max_length:
        return task_name
    return '_\n'.join(task_name.split('_'))

def createDataFrame_distance_accuracy_correlation(file_path):
    # Read the CSV file
    df_correlation = pd.read_csv(file_path, index_col=0)  # Assuming the first column is the index

    # Extract tasks (columns of csv - 1)
    tasks = df_correlation.columns.tolist()

    data_correlation = {
        'Task': [],
        'All-examples Correlation': [],
        'All-examples P-value': [],
        'Positive-examples Correlation': [],
        'Positive-examples P-value': []
    }

    # Extract data for each task
    for task in tasks:
        data_correlation['All-examples Correlation'].append(df_correlation.loc['correlation', task])
        data_correlation['All-examples P-value'].append(df_correlation.loc['p_value', task])
        data_correlation['Positive-examples Correlation'].append(df_correlation.loc['correlation_pos', task])
        data_correlation['Positive-examples P-value'].append(df_correlation.loc['p_value_pos', task])
        data_correlation['Task'].append(task)

    # Creating a DataFrame
    df_correlation = pd.DataFrame(data_correlation)

    # Apply the function to split task names
    df_correlation['Task'] = df_correlation['Task'].apply(lambda x: split_task_name(x))

    return df_correlation

def plot__distance_accuracy_correlation(df : pd.DataFrame):
    # Calculate mean values
    mean_correlation_All = df['All-examples Correlation'].mean()
    mean_p_value_All = df['All-examples P-value'].mean()
    mean_correlation_positive = df['Positive-examples Correlation'].mean()
    mean_p_value_positive = df['Positive-examples P-value'].mean()

    # Melt the DataFrame for easier plotting
    df_melted = df.melt(id_vars=['Task'], 
                        value_vars=['All-examples Correlation', 'Positive-examples Correlation'], 
                        var_name='Type', 
                        value_name='Correlation')

    # Adding P-values to the melted DataFrame
    df_melted['P-value'] = df_melted.apply(lambda row: df.loc[df['Task'] == row['Task'], f"{row['Type'].split()[0]} P-value"].values[0], axis=1)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Task', y='Correlation', hue='Type', palette='tab10')

    # Adding error bars to represent p-values
    for i in range(len(df)):
        plt.errorbar(x=i-0.2, y=df['All-examples Correlation'][i], yerr=df['All-examples P-value'][i], fmt='none', c='black', capsize=5)
        plt.errorbar(x=i+0.2, y=df['Positive-examples Correlation'][i], yerr=df['Positive-examples P-value'][i], fmt='none', c='black', capsize=5)

    # Add horizontal lines for mean correlation and p-value
    plt.axhline(mean_correlation_All, color='blue', alpha=0.3, linestyle='--')
    # Add a point for the mean correlation coefficient with its p-value as an error bar
    plt.errorbar(x=len(df) + 1, y=mean_correlation_All, yerr=mean_p_value_All, fmt='o', color='blue', capsize=5)

    # Add background fill for mean p-value
    plt.axhspan(mean_correlation_All - mean_p_value_All, mean_correlation_All + mean_p_value_All, color='blue', alpha=0.05)
    plt.text(len(df) + 0.8, mean_correlation_All + mean_p_value_All + 0.02, f'{mean_correlation_All:.2f} ± {mean_p_value_All:.2f}', color='blue', fontsize=10, ha='center')


    # Add horizontal lines for mean correlation and p-value - positive
    plt.axhline(mean_correlation_positive, color='orange', alpha=0.7, linestyle='--')
    # Add a point for the mean correlation coefficient with its p-value as an error bar
    plt.errorbar(x=len(df) + 1, y=mean_correlation_positive, yerr=mean_p_value_positive, fmt='o', color='orange', capsize=5)

    # Add background fill for mean p-value
    plt.axhspan(mean_correlation_positive - mean_p_value_positive, mean_correlation_positive + mean_p_value_positive, color='orange', alpha=0.2)
    plt.text(len(df) + 0.8, mean_correlation_positive + mean_p_value_positive + 0.02, f'{mean_correlation_positive:.2f} ± {mean_p_value_positive:.2f}', color='orange', fontsize=10, ha='center', fontweight='bold')


    plt.title('Simple Tasks')
    # plt.title('Medium Tasks')
    # plt.title('Complex Tasks')
    plt.xlabel('')
    plt.ylabel('Correlation Coefficient')
    plt.legend(title='', fontsize='small')

    # Rotate x-axis labels
    plt.xticks(rotation=0, ha='center', fontsize=10)

    # Save the figure
    plt.savefig('img-distance_correlation-simple-tasks.pdf', bbox_inches='tight')
    # plt.savefig('img-distance_correlation-medium-tasks.pdf', bbox_inches='tight')
    # plt.savefig('img-distance_correlation-complex-tasks.pdf', bbox_inches='tight')
    plt.show()

def plot__average_distance_accuracy_correlation():
    # Hard-coded data
    data = {
        'Task': ['Simple Tasks', 'Medium Tasks', 'Complex Tasks'],
        'Correlation Coefficient': [-0.196893773, -0.20676701, -0.346744919],
        'P-Value': [0.228091757, 0.267239157, 0.105870723]
    }

    # Creating a DataFrame
    df = pd.DataFrame(data)

    # Create a bar plot with error bars
    plt.figure(figsize=(5, 5))
    barplot = sns.barplot(x='Task', y='Correlation Coefficient', data=df, capsize=.1, color='lightgray')
    plt.errorbar(x=np.arange(len(df)), y=df['Correlation Coefficient'], yerr=df['P-Value'], fmt='none', c='black', capsize=5)

    # Adjust the font size and alignment of task names
    plt.xticks(ha='center', fontsize=10)

    # Add labels and title
    plt.xlabel('', fontsize=12, fontweight='bold')
    plt.ylabel('', fontsize=12, fontweight='bold')
    plt.title('', fontsize=14, fontweight='bold')

    # Add legend with smaller font size
    plt.legend(fontsize='small')

    # Add text annotations for individual bars
    for i, (corr, p_val) in enumerate(zip(df['Correlation Coefficient'], df['P-Value'])):
        plt.text(i, corr + p_val + 0.02, f'{corr:.2f} ± {p_val:.2f}', color='black', fontsize=10, ha='center')

    # Save the figure
    plt.savefig('img-distance_correlation-average.pdf', bbox_inches='tight')
    # Show the plot
    plt.show()
