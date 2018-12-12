import ray
from ray.tune import run_experiments, grid_search

import models
import environments


if __name__ == "__main__":
    ray.init()
    run_experiments({
        "low-res-car-racing": {
            "run": "A3C",
            "env": "low-res-car-racing-road-label",
            "checkpoint_freq": 100,
            "config": {
                "num_gpus": 0.33,
                # "lr": 0.001,
                # "num_gpus_per_worker": 1/20,
                "num_workers": 3,
                "sample_async": False,
                "env_config": {
                    "resolution": (20, 20), # grid_search([(20, 20), (10, 10)]) #, (6, 6)])
                    },
                "model": {"custom_model": "ReducedKhanElibolModel"},
            },
            "stop" : {
                "episode_reward_mean": 1000
            },
            }
        })
