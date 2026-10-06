import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Конверсия по группам
conv = df.groupby('group')['converted'].mean()
conv.plot(kind='bar', ax=axes[0, 0], color=['steelblue', 'coral'])
axes[0, 0].set_title('Конверсия по группам')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['A (контроль)', 'B (тест)'], rotation=0)
for i, v in enumerate(conv):
    axes[0, 0].text(i, v + 0.002, f'{v:.2%}', ha='center', fontweight='bold')

# 2. Конверсия по устройствам
device = df.groupby(['device', 'group'])['converted'].mean().unstack()
device.plot(kind='bar', ax=axes[0, 1], color=['steelblue', 'coral'])
axes[0, 1].set_title('Конверсия по устройствам')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['Desktop', 'Mobile'], rotation=0)
axes[0, 1].legend(['A (контроль)', 'B (тест)'])
for container in axes[0, 1].containers:
    axes[0, 1].bar_label(container, fmt='%.2f%%', fontsize=9)

# 3. Конверсия по странам
country = df.groupby(['country', 'group'])['converted'].mean().unstack()
country.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'coral'])
axes[1, 0].set_title('Конверсия по странам')
axes[1, 0].set_xticks([0, 1, 2])
axes[1, 0].set_xticklabels(['BY', 'KZ', 'RU'], rotation=0)
axes[1, 0].legend(['A (контроль)', 'B (тест)'])
for container in axes[1, 0].containers:
    axes[1, 0].bar_label(container, fmt='%.2f%%', fontsize=9)

# 4. Длительность сессии
sns.boxplot(x='group', y='session_duration', data=df, ax=axes[1, 1],
            hue='group', palette=['steelblue', 'coral'], legend=False)
axes[1, 1].set_title('Длительность сессии')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['A (контроль)', 'B (тест)'], rotation=0)
medians = df.groupby('group')['session_duration'].median()
for i, v in enumerate(medians):
    axes[1, 1].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('ab_test_result.png', dpi=300)
plt.show()
