
class LogisticDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def ex_kurtosis(self):
        if self.scale <= 0:
            raise Exception("Moment undefined")
        return 24 / 5  # A logisztikus eloszlás többlet csúcsossága mindig 24/5, nem szükséges számítani

#1
def evens_from_list(input_list):
    even_elements = [x for x in input_list if x % 2 == 0]
    return even_elements

#2
def every_element_is_odd(input_list):
    for element in input_list:
        if element % 2 == 0:
            return False
    return True

#3
def kth_largest_in_list(input_list, kth_largest):
    # Ellenőrizzük, hogy a k pozitív és nem nagyobb, mint a lista hossza
    if kth_largest <= 0 or kth_largest > len(input_list):
        raise ValueError("Érvénytelen k érték.")

    # Rendezzük a listát csökkenő sorrendbe
    sorted_list = sorted(input_list, reverse=True)

    # Vegyük az első k legnagyobb elemet
    kth_largest_element = sorted_list[kth_largest - 1]

    return kth_largest_element

#4
def cumavg_list(input_list):
    cumulative_sum = 0
    cumulative_average = []

    for i, value in enumerate(input_list, 1):
        cumulative_sum += value
        average = cumulative_sum / i
        cumulative_average.append(average)

    return cumulative_average

#5
def element_wise_multiplication(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("A két lista hossza nem egyezik meg.")

    result = [x * y for x, y in zip(input_list1, input_list2)]
    return result

#6
def merge_lists(*lists):
    merged_list = []
    for lst in lists:
        merged_list.extend(lst)
    return merged_list

#7
def squared_odds(input_list):
    squared_odds_list = [x**2 for x in input_list if x % 2 != 0]
    return squared_odds_list

#8
def reverse_sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[0], reverse=True))
    return sorted_dict

#9
def sort_list_by_divisibility(input_list):
    result_dict = {
        'by_two': [],
        'by_five': [],
        'by_two_and_five': [],
        'by_none': []
    }

    for num in input_list:
        if num % 2 == 0 and num % 5 == 0:
            result_dict['by_two_and_five'].append(num)
        elif num % 2 == 0:
            result_dict['by_two'].append(num)
        elif num % 5 == 0:
            result_dict['by_five'].append(num)
        else:
            result_dict['by_none'].append(num)

    return result_dict