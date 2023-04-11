
from datetime import datetime


class Timer:
    def __init__(self) -> None:
        self.stopwatches = {}

    def start(self, key):
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        print("Stopwatch <" + key + "> started at " + current_time)

        self.stopwatches[key] = now
    
    def stop(self, key):
        now = datetime.now()

        time_delta = now - self.stopwatches[key]
        current_str = now.strftime("%H:%M:%S")
        delta_str = self._format_timedelta(time_delta)
        print("Stopwatch <" + key + "> stopped at " + current_str + ", time elapsed: " + delta_str + ".")

    def _format_timedelta(self, delta) -> str:
        seconds = int(delta.total_seconds())

        secs_in_a_day = 86400
        secs_in_a_hour = 3600
        secs_in_a_min = 60

        days, seconds = divmod(seconds, secs_in_a_day)
        hours, seconds = divmod(seconds, secs_in_a_hour)
        minutes, seconds = divmod(seconds, secs_in_a_min)

        time_fmt = f"{hours:02d} hours, {minutes:02d} minutes, {seconds:02d} seconds"

        if days > 0:
            suffix = "s" if days > 1 else ""
            return f"{days} day{suffix}, {time_fmt}"

        return time_fmt