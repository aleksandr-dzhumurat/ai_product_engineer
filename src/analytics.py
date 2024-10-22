import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """
    Расчет размера выборки для A/B теста пропорций
    
    Parameters:
    -----------
    baseline_rate : float
        Базовый CTR (конверсия) в группе A
    mde : float
        Минимальный детектируемый эффект (в процентах)
    alpha : float
        Уровень значимости (вероятность ошибки I рода)
    power : float
        Статистическая мощность (1 - β)
    
    Returns:
    --------
    n : int
        Размер выборки для каждой группы
    """
    # Конверсия в группе B
    treatment_rate = baseline_rate * (1 + mde / 100)
    
    # Z-значения
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # двусторонний тест
    z_beta = stats.norm.ppf(power)
    
    # Объединенная пропорция
    p_pooled = (baseline_rate + treatment_rate) / 2
    
    # Дисперсия
    sigma_squared = 2 * p_pooled * (1 - p_pooled)
    
    # Разница
    delta = treatment_rate - baseline_rate
    delta_squared = delta ** 2
    
    # Размер выборки
    n = (z_alpha + z_beta) ** 2 * sigma_squared / delta_squared
    
    return int(np.ceil(n))

def sample_size_demo():
    # Пример из конспекта
    baseline_ctr = 0.15  # 15%
    mde = 20  # 20%

    sample_size = calculate_sample_size(baseline_ctr, mde)

    print("Параметры теста:")
    print(f"  Базовый CTR (группа A): {baseline_ctr * 100}%")
    print(f"  MDE: {mde}%")
    print(f"  Ожидаемый CTR (группа B): {baseline_ctr * (1 + mde/100) * 100}%")
    print("  Уровень значимости (α): 0.05")
    print("  Статистическая мощность (1-β): 0.80")
    print(f"\nНеобходимый размер выборки для каждой группы: {sample_size}")
    print(f"Общий размер выборки (обе группы): {sample_size * 2}")

    # Проверка приближенной формулы n ≈ 160/Δ²
    delta = baseline_ctr * (1 + mde/100) - baseline_ctr
    delta_squared = delta ** 2
    n_approx = 160 / delta_squared

    print("\nПроверка приближенной формулы:")
    print(f"  Δ² = {delta_squared:.6f}")
    print(f"  n ≈ 160/Δ² = {int(n_approx)}")


def mde_evaluation_demo():
    # Базовый CTR
    baseline_ctr = 0.15

    # Различные значения MDE
    mde_values = np.arange(5, 51, 5)
    sample_sizes = []

    for mde in mde_values:
        n = calculate_sample_size(baseline_ctr, mde)
        sample_sizes.append(n)

    # График зависимости размера выборки от MDE
    plt.figure(figsize=(10, 6))
    plt.plot(mde_values, sample_sizes, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('MDE (%)', fontsize=12)
    plt.ylabel('Размер выборки (n)', fontsize=12)
    plt.title('Зависимость размера выборки от минимального детектируемого эффекта', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=sample_sizes[3], color='r', linestyle='--', alpha=0.5, 
                label=f'MDE=20%: n={sample_sizes[3]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sample_size_vs_mde.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Таблица результатов
    print("\nТаблица: Размер выборки для разных MDE")
    print("=" * 50)
    print(f"{'MDE (%)':<10} {'CTR_B (%)':<15} {'n (на группу)':<15} {'n (всего)':<15}")
    print("=" * 50)
    for mde, n in zip(mde_values, sample_sizes):
        ctr_b = baseline_ctr * (1 + mde/100)
        print(f"{mde:<10} {ctr_b*100:<15.2f} {n:<15} {n*2:<15}")

def statistics_distribution_demo():
    # Параметры
    alpha = 0.05  # уровень значимости
    z_alpha = stats.norm.ppf(1 - alpha/2)  # для двустороннего теста

    # Создание данных для графиков
    x = np.linspace(-4, 4, 1000)

    # Null distribution (H0)
    null_dist = stats.norm.pdf(x, loc=0, scale=1)

    # Alternative distribution (H1)
    effect_size = 1.5  # размер эффекта
    alt_dist = stats.norm.pdf(x, loc=effect_size, scale=1)

    # Построение графика
    plt.figure(figsize=(12, 6))

    # Нулевое распределение
    plt.plot(x, null_dist, 'b-', linewidth=2, label='Null distribution (H₀)')
    plt.fill_between(x[x > z_alpha], null_dist[x > z_alpha], alpha=0.3, color='red', 
                    label=f'α = {alpha} (Type I Error)')

    # Альтернативное распределение
    plt.plot(x, alt_dist, 'g-', linewidth=2, label='Alternative distribution (H₁)')
    plt.fill_between(x[x < z_alpha], alt_dist[x < z_alpha], alpha=0.3, color='orange',
                    label='β (Type II Error)')
    plt.fill_between(x[x > z_alpha], alt_dist[x > z_alpha], alpha=0.3, color='green',
                    label='Power = 1-β')

    # Критическая граница
    plt.axvline(z_alpha, color='red', linestyle='--', linewidth=2, label=f'Critical value = {z_alpha:.2f}')

    plt.xlabel('Test Statistic', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('A/B Test: Type I Error (α), Type II Error (β), and Statistical Power', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ab_test_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Уровень значимости (α): {alpha}")
    print(f"Критическое значение: {z_alpha:.3f}")
    print(f"Размер эффекта: {effect_size}")