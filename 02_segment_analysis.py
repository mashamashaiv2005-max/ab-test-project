import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('ab_test.csv')

# 1. Общая конверсия
print("КОНВЕРСИЯ")
conv = df.groupby('group')['converted'].mean()
print(conv)
print(f"Разница: {conv['B'] - conv['A']:.4f}")

# 2. Конверсия по устройствам
print("\nКОНВЕРСИЯ ПО УСТРОЙСТВАМ")
device_conv = df.groupby(['device', 'group'])['converted'].mean().unstack()
print(device_conv)
print(f"\nРазница по desktop: {device_conv.loc['desktop', 'B'] - device_conv.loc['desktop', 'A']:.4f}")
print(f"Разница по mobile: {device_conv.loc['mobile', 'B'] - device_conv.loc['mobile', 'A']:.4f}")

# 3. Конверсия по странам
print("\nКОНВЕРСИЯ ПО СТРАНАМ")
country_conv = df.groupby(['country', 'group'])['converted'].mean().unstack()
country_conv['Разница'] = country_conv['B'] - country_conv['A']
print(country_conv)

# 4. Длительность сессии
print("\nДЛИТЕЛЬНОСТЬ СЕССИИ")
session = df.groupby('group')['session_duration'].mean()
print(session)
print(f"Разница: {session['B'] - session['A']:.2f} сек")

# 5. t-тест по длительности
session_a = df[df['group'] == 'A']['session_duration']
session_b = df[df['group'] == 'B']['session_duration']
t_stat_s, p_value_s = stats.ttest_ind(session_a, session_b)
print(f"t-тест: t = {t_stat_s:.4f}, p-value = {p_value_s:.4f}")

if p_value_s < 0.05:
    print("Длительность сессии значимо изменилась.")
else:
    print("Длительность сессии не изменилась значимо.")

# 6. Z-тест по сегментам
print("\nZ-ТЕСТ ПО СЕГМЕНТАМ")

def z_test_for_segment(data, segment_col, segment_val):
    subset = data[data[segment_col] == segment_val]

    n_a = len(subset[subset['group'] == 'A'])
    n_b = len(subset[subset['group'] == 'B'])

    x_a = subset[subset['group'] == 'A']['converted'].sum()
    x_b = subset[subset['group'] == 'B']['converted'].sum()

    p_a = x_a / n_a
    p_b = x_b / n_b

    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return z, p_value, p_a, p_b

# По устройствам
for device in ['desktop', 'mobile']:
    z, p, pa, pb = z_test_for_segment(df, 'device', device)
    print(f"\n{device}:")
    print(f"  Конверсия A = {pa:.4f}, B = {pb:.4f}")
    print(f"  Z = {z:.4f}, p-value = {p:.6f}")
    if p < 0.05:
        print(f"Значимо")
    else:
        print(f"Не значимо")

# По странам
for country in ['RU', 'KZ', 'BY']:
    z, p, pa, pb = z_test_for_segment(df, 'country', country)
    print(f"\n{country}:")
    print(f"  Конверсия A = {pa:.4f}, B = {pb:.4f}")
    print(f"  Z = {z:.4f}, p-value = {p:.6f}")
    if p < 0.05:
        print(f"Значимо")
    else:
        print(f"Не значимо")
