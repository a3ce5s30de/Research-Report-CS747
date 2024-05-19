import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
data = {
    'Topics': [i for i in range(10)],
    'Strongly Disagree': [11, 10, 6, 2, 13, 8, 5, 8, 10, 4],
    'Disagree': [17, 18, 6, 15, 34, 10, 6, 19, 9, 9],
    'Neutral': [46, 29, 22, 17, 29, 28, 38, 16, 26, 18],
    'Agree': [38, 15, 36, 45, 46, 33, 18, 21, 32, 19],
    'Strongly Agree': [14, 7, 5, 2, 7, 4, 7, 7, 3, 4]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 8))
sns.lineplot(data=df.set_index('Topics'), dashes=False, markers=True)
plt.title('Topic Distribution Across Different Levels of Enjoyment')
plt.xlabel('Topic Number')
plt.ylabel('Number of Students')
plt.grid(True)
plt.xticks(range(0, 10))  # Ensure x-axis labels are from 0 to 9
plt.legend(title='Enjoyment Levels')
plt.show()

