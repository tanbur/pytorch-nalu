

def num_to_text(num):
    # Make sure we have an integer
    """
    Given a number, write out the English representation of it.
    :param num:
    :return:

    >>> num_to_text(3)
    'three'
    >>> num_to_text(14)
    'fourteen'
    >>> num_to_text(24)
    'twenty four'
    >>> num_to_text(31)
    'thirty one'
    >>> num_to_text(49)
    'fourty nine'
    >>> num_to_text(56)
    'fifty six'
    >>> num_to_text(156)
    'one hundred and fifty six'
    >>> num_to_text(700)
    'seven hundred'
    >>> num_to_text(999)
    'nine hundred and ninety nine'
    >>> num_to_text(123456)
    'one hundred and twenty three thousand four hundred and fifty six'

    >>> num_to_text(123456789)
    'one hundred and twenty three million four hundred and fifty six thousand seven hundred and eighty nine'
    >>> num_to_text(123456789000)
    'one hundred and twenty three billion four hundred and fifty six million seven hundred and eighty nine thousand'
    >>> num_to_text(12000000000000000)
    'twelve million billion'
    >>> num_to_text(12000000000000000000000)
    'twelve thousand billion billion'
    >>> num_to_text(-79)
    'negative seventy nine'
    """
    num = int(num)

    BASE_CASES = {i: word for i, word in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                                                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen'])}
    BASE_CASES.update({15: 'fifteen', 20: 'twenty', 30: 'thirty', 50: 'fifty', 80: 'eighty'})

    def rem_zero(str_):
        if str_.endswith(' and zero'):
            return str_[:-9]
        elif str_.endswith(' zero'):
            return str_[:-5]
        else:
            return str_

    # Handle negative numbers
    if num < 0:
        return 'negative {}'.format(num_to_text(- num))

    name = BASE_CASES.get(num)
    if name is not None:
        return name

    numstr = str(num)
    if len(numstr) == 2:
        # Teens are special case
        if numstr[0] == '1':
            return num_to_text(numstr[1]) + 'teen'
        elif numstr[1] == '0':
            # We're not a teen and we're not a base case, so we must be x0 for x in [4, 6, 7, 9]
            return num_to_text(numstr[0]) + 'ty'
        else:
            return num_to_text(numstr[0] + '0') + ' ' + num_to_text(numstr[1])

    if len(numstr) == 3:
        return rem_zero('{} hundred and {}'.format(num_to_text(numstr[0]), num_to_text(numstr[1:])))

    # Sort out the thousands and billions
    if len(numstr) > 9:
        return rem_zero(num_to_text(numstr[:-9]) + ' billion ' + num_to_text(numstr[-9:]))
    elif len(numstr) > 6:
        return rem_zero(num_to_text(numstr[:-6]) + ' million ' + num_to_text(numstr[-6:]))
    elif len(numstr) > 3:
        return rem_zero(num_to_text(numstr[:-3]) + ' thousand ' + num_to_text(numstr[-3:]))
    return 'ERROR'


if __name__ == '__main__':
    import doctest
    doctest.testmod()
