def sum_digits(n):
    n = abs(n)  
    digit_sum = 0
    while n > 0:
        digit_sum += n % 10
        n //= 10
    return digit_sum

def diff_sum_digits(n):
    n = abs(n)
    return n - sum_digits(n)

def wraps_diff_sum_digits(n):
    while n >= 10:  
        n = diff_sum_digits(n)
    return n