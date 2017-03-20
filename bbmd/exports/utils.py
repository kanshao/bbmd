
def check_valid_flat_format(fmt):
    valid_fmt = ['txt', 'xlsx']
    if fmt not in valid_fmt:
        raise ValueError('Invalid format: expecting {}'.format('|'.join(valid_fmt)))
    else:
        return True


def check_valid_report_format(fmt):
    valid_fmt = ['txt', 'json']
    if fmt not in valid_fmt:
        raise ValueError('Invalid format: expecting {}'.format('|'.join(valid_fmt)))
    else:
        return True
