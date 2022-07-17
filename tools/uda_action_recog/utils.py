

import getpass


def check_username():
    if getpass.getuser() == 'eicg':
        env_id = 0
    elif getpass.getuser() == 'lin':
        env_id = 1
    else:
        raise Exception("Unknown username!")
    return env_id