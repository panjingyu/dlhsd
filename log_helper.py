"""Logging helper class."""


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