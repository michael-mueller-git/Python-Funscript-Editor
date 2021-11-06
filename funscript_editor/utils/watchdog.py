from threading import Timer

class Watchdog(Exception):
    def __init__(self, timeout_in_seconds, userHandler=None):
        self.timeout = timeout_in_seconds
        self.handler = userHandler if userHandler is not None else self.defaultHandler
        self.timer = Timer(self.timeout, self.handler)
        self.started = False

    def start(self):
        if not self.started:
            self.started = True
            self.timer.start()

    def trigger(self):
        self.timer.cancel()
        self.timer = Timer(self.timeout, self.handler)
        self.timer.start()

    def stop(self):
        try: self.timer.cancel()
        except: pass

    def defaultHandler(self):
        raise self
