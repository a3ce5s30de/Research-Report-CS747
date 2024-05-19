import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
difficulty_data = {
    'Topics': [i for i in range(10)],
    'Strongly Disagree': [1, 1, 2, 2, 1, 0, 1, 4, 1, 2],  # Note the addition of 0 for missing topic 5
    'Disagree': [11, 6, 5, 7, 6, 1, 7, 6, 8, 5],
    'Neutral': [28, 9, 29, 15, 27, 28, 32, 25, 8, 26],
    'Agree': [49, 24, 29, 38, 59, 47, 31, 21, 36, 34],
    'Strongly Agree': [30, 12, 13, 16, 16, 21, 13, 18, 20, 21]
}

# Create DataFrame
df = pd.DataFrame(difficulty_data)

# Plotting
plt.figure(figsize=(12, 8))
sns.lineplot(data=df.set_index('Topics'), dashes=False, markers=True)
plt.title('Topic Distribution Across Different Levels of Difficulty')
plt.xlabel('Topic Number')
plt.ylabel('Number of Students')
plt.grid(True)
plt.xticks(range(0, 10))  # Ensure x-axis labels are from 0 to 9
plt.legend(title='Enjoyment Levels')
plt.show()