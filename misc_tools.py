def element_count(p):
    elements = list(p)
    count = 0
    while elements:
        entry = elements.pop()
        count += 1
        if isinstance(entry, list):
            elements.extend(entry)
    return count
