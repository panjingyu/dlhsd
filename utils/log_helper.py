"""Logging helper class."""


import logging
import sys


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level, tee=None):
       self.logger = logger
       self.level = level
       self.linebuf = ''
       self.tee = tee

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())
          if self.tee is not None:
              self.tee.write(line)
              self.tee.write('\n')

    def flush(self):
        if self.tee is not None:
            self.tee.flush()

def make_logger(log_file, log_stdin=False):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s:%(message)s',
        filename=log_file,
        filemode='w'
        )
    logger = logging.getLogger('')
    if log_stdin:
        sys.stdout = StreamToLogger(logger, logging.INFO, sys.stdout)
    return logger

def get_log_id(args, delim='.'):
    log_id = []
    log_id.append(f'lr{args.lr:.1e}')
    if args.cure_l != 0. and args.cure_h != 0.:
        log_id.append('cureL{:.1e}H{:.1e}'.format(args.cure_l, args.cure_h))
    return delim.join(log_id)
