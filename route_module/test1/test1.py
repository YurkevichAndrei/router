from pathlib import Path
from datetime import datetime
import time

k_horiz_v = 0.35
k_vert_v = 1.0


def p_path():
    start_time = datetime.now()
    time.sleep(1)
    print(Path(f"{Path.cwd()}\\files_h_{k_horiz_v}_v_{k_vert_v}\\"))
    print(datetime.now() - start_time)
