from __future__ import annotations

import json

from ola_support_agent.demo import run_demo


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))

