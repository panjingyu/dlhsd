from datetime import datetime

def get_timestamp(fmt='%Y%m%d%H%M%S'):
    return datetime.utcnow().strftime(fmt)

if __name__ == '__main__':
    print('get_timestamp:', get_timestamp())
