#1
import pandas as pd
file_path = '/Users/molnartamas07/Documents/ECOPY_23241/data/chipotle.tsv'
# Próba
try:
    food = pd.read_csv(file_path, sep='\t')
    print("sikeres.")
except FileNotFoundError:
    print("nem található.")
except Exception as e:
    print(f"Hiba: {e}")
#2
def change_price_to_float(input_df):
    # Másolat készítése a bemeneti DataFrame-ről
    df_copy = input_df.copy()

    # Az "item_price" oszlopból távolítsa el a dollárjelet és átalakítsa float-tá
    df_copy['item_price'] = df_copy['item_price'].str.replace('$', '').astype(float)

    return df_copy

#3
def number_of_observations(input_df):
    num_observations = input_df.shape[0]

    return num_observations

#4
def items_and_prices(input_df):
    item_price_df = input_df[['item_name', 'item_price']]

    return item_price_df

#5
def sorted_by_price(input_df):
    # Az ár szerint csökkenő sorrendben rendezett DataFrame létrehozása
    sorted_df = input_df.sort_values(by='item_price', ascending=False)

    return sorted_df

#6
def avg_price(input_df):
    average_price = input_df['item_price'].mean()

    return average_price

#7

def unique_items_over_ten_dollars(input_df):
    # Szűrjük azokat az elemeket, amelyek 10 dollárnál drágábbak
    filtered_df = input_df[input_df['item_price'] > 10]

    # Az egyedi elemek kiválasztása (név, feltét és ár alapján)
    unique_filtered_df = filtered_df.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])

    return unique_filtered_df

#8

def items_starting_with_s(input_df):
    # Szűrjük azokat a sorokat, ahol a "item_name" oszlop 'S'-el kezdődik
    filtered_df = input_df[input_df['item_name'].str.startswith('S')]

    return filtered_df

#9
def first_three_columns(input_df):
    selected_columns = input_df.iloc[:, :3]

    return selected_columns

#10
def every_column_except_last_two(input_df):
    # Az összes oszlop kiválasztása az utolsó két oszlop nélkül
    selected_columns = input_df.iloc[:, :-2]

    return selected_columns

#11

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    # Szűrjük azokat a sorokat, ahol a megadott oszlop értéke megegyezik a megadott értékek egyikével
    filtered_df = input_df[input_df[column_to_filter].isin(rows_to_keep)]

    # Csak a megadott oszlopokat tartjuk meg
    result_df = filtered_df[columns_to_keep]

    return result_df


#12

def generate_quartile(input_df):
    # Függvény a kvartilis érték kiszámításához az "item_price" oszlopból
    def determine_quartile(price):
        if price >= 30:
            return 'premium'
        elif price >= 20:
            return 'high-cost'
        elif price >= 10:
            return 'medium-cost'
        else:
            return 'low-cost'

    # Az "item_price" oszlop alapján meghatározzuk a kvartilist és hozzáadjuk az új oszlopot
    input_df['Quartile'] = input_df['item_price'].apply(determine_quartile)

    return input_df


#13
def average_price_in_quartiles(input_df):
    # Csoportosítás a "Quartile" oszlop alapján és az átlagos ár kiszámítása
    quartile_avg_prices = input_df.groupby('Quartile')['item_price'].mean()
    return quartile_avg_prices


#14
def minmaxmean_price_in_quartile(input_df):
    # Csoportosítás a "Quartile" oszlop alapján és a minimális, maximális és átlagos árak kiszámítása
    quartile_price_summary = input_df.groupby('Quartile')['item_price'].agg(['min', 'max', 'mean'])
    return quartile_price_summary


#15
from typing import List

def gen_uniform_mean_trajectories(distribution: tuple, number_of_trajectories: int, length_of_trajectory: int) -> List[
    List[float]]:
    # Azonos seed beállítása a reprodukálhatóság érdekében
    random.seed(42)

    # Ebben a listában tároljuk az eredményeket
    result = []

    for _ in range(number_of_trajectories):
        # Minden egyes új belső lista
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            value = random.uniform(distribution[0], distribution[1])
            cumulative_sum += value
            trajectory.append(cumulative_sum / (len(trajectory) + 1))

        result.append(trajectory)

    return result


#16
def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Azonos seed beállítása a reprodukálhatóság érdekében
    random.seed(42)

    # Ebben a listában tároljuk az eredményeket
    result = []

    for _ in range(number_of_trajectories):
        # Minden egyes új belső lista
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            value = random.gauss(distribution[0], distribution[1])
            cumulative_sum += value
            trajectory.append(cumulative_sum / (len(trajectory) + 1))

        result.append(trajectory)

    return result


#17
def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Azonos seed beállítása a reprodukálhatóság érdekében
    random.seed(42)

    # Ebben a listában tároljuk az eredményeket
    result = []

    for _ in range(number_of_trajectories):
        # Minden egyes új belső lista
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            value = random.gauss(distribution[0], distribution[1])
            cumulative_sum += value
            trajectory.append(cumulative_sum / (len(trajectory) + 1))

        result.append(trajectory)

    return result


#18
def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Azonos seed beállítása a reprodukálhatóság érdekében
    random.seed(42)

    # Ebben a listában tároljuk az eredményeket
    result = []

    for _ in range(number_of_trajectories):
        # Minden egyes új belső lista
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            # A Cauchy-eloszlás szimulálása normál eloszlás alapján
            value = random.gauss(distribution[0], distribution[1])
            cumulative_sum += value
            trajectory.append(cumulative_sum / (len(trajectory) + 1))

        result.append(trajectory)

    return result


#19
import random
def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    # Azonos seed beállítása a reprodukálhatóság érdekében
    random.seed(42)

    # Ebben a listában tároljuk az eredményeket
    result = []

    for _ in range(number_of_trajectories):
        # Minden egyes új belső lista
        trajectory = []
        cumulative_sum = 0.0

        for _ in range(length_of_trajectory):
            # Számold ki az alpha és beta értékeket a `ChiSquaredDistribution` alapján
            alpha = distribution.degrees_of_freedom  # Az alakparaméter a szabadsági fokok számával egyenlő
            beta = 2.0  # Általában a skálázási paraméter értéke 2 a chi-négyzet eloszlás esetén

            # Generálj gamma eloszlású véletlen értéket
            value = random.gammavariate(alpha, beta)

            cumulative_sum += value
            trajectory.append(cumulative_sum / (len(trajectory) + 1))

        result.append(trajectory)

    return result
