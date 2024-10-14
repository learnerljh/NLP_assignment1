def flatten_list(nested_list: list):
    flattened_list = []
    for item in nested_list:
        for i in item:
            flattened_list.append(i)
    return flattened_list

def char_count(s: str):
    count = {}
    for char in s:
        if char in count:
            count[char] += 1
        else:
            count[char] = 1
    return count

