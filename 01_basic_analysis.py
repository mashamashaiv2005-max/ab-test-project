import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('ab_test.csv')

group_a = df[df['group'] == 'A']['converted']
group_b = df[df['group'] == 'B']['converted']

# Базовые метрики
n_a = len(group_a)
n_b = len(group_b)

x_a = group_a.sum()
x_b = group_b.sum()

p_a = x_a / n_a
p_b = x_b / n_b

diff = p_b - p_a
rel_growth = (p_b / p_a - 1) * 100

print(f"Конверсия A: {p_a:.4f} ({p_a*100:.2f}%)")
print(f"Конверсия B: {p_b:.4f} ({p_b*100:.2f}%)")
print(f"Разница: {diff:.4f} ({diff*100:.2f} %)")
print(f"Относительный рост: {rel_growth:.2f}%")

# Z-тест
p_pool = (x_a + x_b) / (n_a + n_b)
se_z = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
z_stat = (p_b - p_a) / se_z
p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))

z_crit = 1.96
ci_lower_z = diff - z_crit * se_z
ci_upper_z = diff + z_crit * se_z

print(f"\nZ-тест:")
print(f"Z = {z_stat:.4f}, p-value = {p_value_z:.6f}")
print(f"95% CI: [{ci_lower_z:.4f}; {ci_upper_z:.4f}]")

# t-тест
t_stat, p_value_t = stats.ttest_ind(group_a, group_b)

var_a = group_a.var()
var_b = group_b.var()
se_t = np.sqrt(var_a/n_a + var_b/n_b)

t_crit = stats.t.ppf(0.975, df=n_a + n_b - 2)
ci_lower_t = diff - t_crit * se_t
ci_upper_t = diff + t_crit * se_t

print(f"\nt-тест:")
print(f"t = {t_stat:.4f}, p-value = {p_value_t:.6f}")
print(f"95% CI: [{ci_lower_t:.4f}; {ci_upper_t:.4f}]")

# Сравнение
comparison = pd.DataFrame({
    'Метрика': ['Статистика', 'p-value', 'CI нижняя', 'CI верхняя'],
    'Z-тест': [f'{z_stat:.4f}', f'{p_value_z:.6f}', f'{ci_lower_z:.4f}', f'{ci_upper_z:.4f}'],
    't-тест': [f'{t_stat:.4f}', f'{p_value_t:.6f}', f'{ci_lower_t:.4f}', f'{ci_upper_t:.4f}']
})
print(f"\n{comparison}")

# Вывод
alpha = 0.05
if p_value_z < alpha and ci_lower_z > 0:
    print("\nZ-тест: разница значима.")
else:
    print("\nZ-тест: разница не значима.")

if p_value_t < alpha and ci_lower_t > 0:
    print("t-тест: разница значима.")
else:
    print("t-тест: разница не значима.")
