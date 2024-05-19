import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Load the data, skipping the second and third rows for the grades
grades = pd.read_excel('FinalGrades.xlsx', skiprows=[1, 2])
responses = pd.read_excel('Lab7Responses.xlsx')

# Map Likert scale responses to numerical values
likert_mapping = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly agree': 5
}
responses['Enjoyable'] = responses['Response 1'].map(likert_mapping)
responses['Difficult'] = responses['Response 2'].map(likert_mapping)

# Merge data on ANON_ID, assuming ANON_ID is properly formatted in both datasets
data = pd.merge(responses[['ANON_ID', 'Enjoyable', 'Difficult']], grades[['ANON_ID', 'Final Score']], on='ANON_ID')

# Convert ANON_ID to integer type to remove decimal part
data['ANON_ID'] = data['ANON_ID'].astype(int)

# Group ANON_ID by enjoyment and difficulty levels
grouped_by_enjoyment = data.groupby('Enjoyable')['ANON_ID'].apply(list)
grouped_by_difficulty = data.groupby('Difficult')['ANON_ID'].apply(list)

# Export to .txt files without conversion to string DataFrame representation
def save_ids_to_file(filename, grouped_data):
    with open(filename, 'w') as file:
        for key, ids in grouped_data.items():
            file.write(f"Level {key}:\n")
            file.write(', '.join(map(str, ids)) + '\n\n')

save_ids_to_file('grouped_by_enjoyment.txt', grouped_by_enjoyment)
save_ids_to_file('grouped_by_difficulty.txt', grouped_by_difficulty)


# Visualization of final scores by enjoyment and difficulty using colorful palettes
plt.figure(figsize=(12, 6))
sns.boxplot(x='Enjoyable', y='Final Score', data=data, hue='Enjoyable', palette='coolwarm', dodge=False)
plt.title('Final Score by Enjoyment of Programming')
plt.xlabel('Level of Enjoyment')
plt.ylabel('Final Score')
plt.xticks([0, 1, 2, 3, 4], ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'])
plt.legend([])
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Difficult', y='Final Score', data=data, hue='Difficult', palette='coolwarm', dodge=False)
plt.title('Final Score by Perceived Difficulty of Programming')
plt.xlabel('Perceived Difficulty')
plt.ylabel('Final Score')
plt.xticks([0, 1, 2, 3, 4], ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'])
plt.legend([])
plt.grid(True)
plt.tight_layout()
plt.show()

# Spearman rank correlation for enjoyment and final scores
spearman_enjoyment, p_value_enjoyment = spearmanr(data['Enjoyable'], data['Final Score'])
print(f"Spearman correlation between enjoyment and final scores: {spearman_enjoyment:.3f}, p-value: {p_value_enjoyment:.3f}")

# Spearman rank correlation for difficulty and final scores
spearman_difficulty, p_value_difficulty = spearmanr(data['Difficult'], data['Final Score'])
print(f"Spearman correlation between difficulty and final scores: {spearman_difficulty:.3f}, p-value: {p_value_difficulty:.3f}")
