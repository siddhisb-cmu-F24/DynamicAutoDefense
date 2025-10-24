#!/usr/bin/env python3
import os
from os.path import join, exists, dirname, basename, splitext
from joblib import Parallel, delayed
import pandas as pd
from glob import glob
from tqdm import tqdm
import time
import json
import statistics
import requests

from defense.explicit_detector.agency.explicit_1_agent import (
    VanillaJailbreakDetector, CoT, CoTV2, CoTV3, VanillaJailbreakDetectorV0125
)
from defense.explicit_detector.agency.explicit_2_agents import AutoGenDetectorV1, AutoGenDetectorV0125
from defense.explicit_detector.agency.explicit_3_agents import AutoGenDetectorThreeAgency, AutoGenDetectorThreeAgencyV2
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
# from defense.implicit_detector.agency.implicit_1_agent import MoralAdvisor
# from defense.implicit_detector.agency.implicit_2_agents import MoralAdvisor2Agent
# from defense.implicit_detector.agency.implicit_3_agents import MoralAdvisor3Agent
# from defense.implicit_detector.implicit_defense_arch import ImplicitMultiAgentDefense
from evaluator.evaluate_helper import evaluate_defense_with_output_list, evaluate_defense_with_response
import argparse

defense_strategies = [
    # {"name": "im-1", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor},
    # {"name": "im-2", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor2Agent},
    # {"name": "im-3", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor3Agent},
    # {"name": "ex-1", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetector},
    # {"name": "ex-2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV1},
    {"name": "ex-3", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgency},
    # {"name": "ex-cot", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoT},
    # {"name": "ex-1-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetectorV0125},
    # {"name": "ex-2-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV0125},
    # {"name": "ex-cot-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV3},
    # {"name": "ex-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": DetectorFiveAgency},
    # {"name": "ex-3-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgencyV2},
    # {"name": "ex-cot-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV2},
]

# ---------------- Latency instrumentation ----------------
class _LatencyRecorder:
    def __init__(self):
        self.enabled = False
        self._orig_post = None
        self.per_call_ms = []   # raw per-call latencies (ms)
        self.run_wall_start = None
        self.run_wall_end = None

    def _timed_post(self, *args, **kwargs):
        t0 = time.perf_counter()
        resp = self._orig_post(*args, **kwargs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.per_call_ms.append(dt_ms)
        return resp

    def start(self):
        if self.enabled:
            return
        self.enabled = True
        # monkey-patch requests.post for timing ALL LLM calls within this process
        self._orig_post = requests.post
        requests.post = self._timed_post
        self.run_wall_start = time.perf_counter()

    def stop(self):
        if not self.enabled:
            return
        self.run_wall_end = time.perf_counter()
        # restore
        requests.post = self._orig_post
        self._orig_post = None
        self.enabled = False

    @property
    def wall_time_sec(self):
        if self.run_wall_start is None or self.run_wall_end is None:
            return None
        return self.run_wall_end - self.run_wall_start

    def summary(self):
        calls = len(self.per_call_ms)
        if calls == 0:
            return {
                "llm_calls": 0,
                "per_call_avg_ms": None,
                "per_call_p50_ms": None,
                "per_call_p95_ms": None,
            }
        pcs = sorted(self.per_call_ms)
        avg = sum(pcs) / calls
        p50 = pcs[int(0.5 * (calls - 1))]
        p95 = pcs[int(0.95 * (calls - 1))]
        return {
            "llm_calls": calls,
            "per_call_avg_ms": avg,
            "per_call_p50_ms": p50,
            "per_call_p95_ms": p95,
        }

LAT = _LatencyRecorder()
# --------------------------------------------------------

def _append_latency_summary_row(csv_path, row_dict):
    df_row = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(csv_path, index=False)

def _compute_num_samples(output_json_path):
    try:
        with open(output_json_path, "r") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 1
    except Exception:
        return None

def eval_csv_from_yuan():
    attack_csv_list = glob("data/harmful_output/multiple_attack_output/*.csv")
    attack_csv_list.sort()
    for attack_csv in tqdm(attack_csv_list):
        df = pd.read_csv(attack_csv)
        for defense_strategy in defense_strategies:
            df[defense_strategy["name"]] = evaluate_defense_with_output_list(
                task_agency=defense_strategy["task_agency"],
                defense_agency=defense_strategy["defense_agency"],
                output_list=df["target"].tolist())
        df.to_csv(attack_csv, index=False)

def eval_defense_strategies(llm_name, output_suffix, ignore_existing=True,
                            chat_file="data/harmful_output/attack_gpt3.5_1106.json",
                            host_name="127.0.0.1", port_range=(9005, 9005),
                            frequency_penalty=1.3, num_of_threads=6, temperature=0.7, presence_penalty=0.0,
                            measure_latency=False):
    defense_output_prefix = join(f"data/defense_output/open-llm-defense{output_suffix}", llm_name)
    os.makedirs(defense_output_prefix, exist_ok=True)
    latency_csv_path = join(defense_output_prefix, "latency_summary.csv")

    for defense_strategy in defense_strategies:
        output_file = join(defense_output_prefix, defense_strategy["name"] + ".json")
        if exists(output_file) and ignore_existing:
            print("Defense output exists, skip", output_file)
            continue

        print("Evaluating", llm_name, defense_strategy["name"], "\nOutput to", output_file)

        # ---- start latency capture (optional) ----
        if measure_latency:
            LAT.per_call_ms = []
            LAT.start()

        # run the defense
        evaluate_defense_with_response(
            task_agency=defense_strategy["task_agency"],
            defense_agency=defense_strategy["defense_agency"],
            chat_file=chat_file,
            defense_output_name=join(defense_output_prefix, defense_strategy["name"] + ".json"),
            model_name=llm_name,
            port_range=port_range,
            host_name=host_name,
            parallel=True, num_of_threads=num_of_threads,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
            temperature=temperature)

        # ---- stop latency capture and summarize ----
        if measure_latency:
            LAT.stop()
            wall_s = LAT.wall_time_sec or 0.0
            per_call = LAT.summary()
            num_samples = _compute_num_samples(output_file) or 0
            per_sample_avg_ms = (wall_s * 1000.0 / num_samples) if num_samples else None

            # Write a per-run JSON with details
            detail = {
                "llm_name": llm_name,
                "defense_name": defense_strategy["name"],
                "chat_file": chat_file,
                "output_file": output_file,
                "num_samples": num_samples,
                "wall_time_sec": wall_s,
                "per_sample_avg_ms": per_sample_avg_ms,
                **per_call,
            }
            detail_path = join(defense_output_prefix, f"{defense_strategy['name']}_latency.json")
            with open(detail_path, "w") as f:
                json.dump(detail, f, indent=2)

            # Append/Write a CSV summary row
            csv_row = {
                "llm_name": llm_name,
                "defense_name": defense_strategy["name"],
                "chat_file": chat_file,
                "output_file": output_file,
                "num_samples": num_samples,
                "wall_time_sec": round(wall_s, 3),
                "per_sample_avg_ms": round(per_sample_avg_ms, 1) if per_sample_avg_ms is not None else None,
                "llm_calls": per_call["llm_calls"],
                "per_call_avg_ms": round(per_call["per_call_avg_ms"], 1) if per_call["per_call_avg_ms"] else None,
                "per_call_p50_ms": round(per_call["per_call_p50_ms"], 1) if per_call["per_call_p50_ms"] else None,
                "per_call_p95_ms": round(per_call["per_call_p95_ms"], 1) if per_call["per_call_p95_ms"] else None,
            }
            _append_latency_summary_row(latency_csv_path, csv_row)

            print("\nLatency summary")
            for k, v in csv_row.items():
                print(f"  {k}: {v}")

def eval_with_open_llms(model_list, chat_file, port_range=(9005, 9005 + 3), ignore_existing=True,
                        host_name="127.0.0.1", output_suffix="", frequency_penalty=1.3,
                        temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0,
                        measure_latency=False):
    # "llama-2-13b", "llama-2-7b", "llama-pro-8b", "llama-2-70b", "tinyllama-1.1b", "vicuna-13b-v1.5", "vicuna-33b", "vicuna-7b-v1.5", "vicuna-13b-v1.3.0"
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file,
                                    host_name=host_name, port_range=port_range, presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature,
                                    measure_latency=measure_latency)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack", "safe"),
                                    host_name=host_name, port_range=port_range, presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature,
                                    measure_latency=measure_latency)

def eval_with_openai(model_list, chat_file, ignore_existing=True, output_suffix="",
                     temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0,
                     measure_latency=False):
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file, presence_penalty=presence_penalty,
                                    num_of_threads=2, temperature=temperature,
                                    measure_latency=measure_latency)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack_gpt3.5", "safe_gpt3.5"),
                                    presence_penalty=presence_penalty,
                                    num_of_threads=2, temperature=temperature,
                                    measure_latency=measure_latency)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", nargs="*", default=["gpt-3.5-turbo-1106"])
    parser.add_argument("--chat_file", type=str, default="data/harmful_output/attack_gpt3.5_1106.json")
    parser.add_argument("--port_start", type=int, default=9005)
    parser.add_argument("--host_name", nargs="*", default=["127.0.0.1"])
    parser.add_argument("--num_of_instance", type=int, default=1)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--eval_harm", action="store_true")
    parser.add_argument("--eval_safe", action="store_true")
    parser.add_argument("--measure_latency", action="store_true", help="Record wall time, per-call latency p50/p95, etc.")
    args = parser.parse_args()

    if args.model_list[0].startswith("gpt"):
        eval_with_openai(model_list=args.model_list, output_suffix=args.output_suffix, ignore_existing=True,
                         temperature=args.temperature, chat_file=args.chat_file, eval_safe=args.eval_safe,
                         eval_harm=args.eval_harm, presence_penalty=args.presence_penalty,
                         measure_latency=args.measure_latency)
    else:
        port_range = (args.port_start, args.port_start + args.num_of_instance - 1)
        eval_with_open_llms(model_list=args.model_list, port_range=port_range, ignore_existing=True,
                            output_suffix=args.output_suffix, host_name=args.host_name,
                            frequency_penalty=args.frequency_penalty, temperature=args.temperature,
                            chat_file=args.chat_file, eval_safe=args.eval_safe, eval_harm=args.eval_harm,
                            presence_penalty=args.presence_penalty, measure_latency=args.measure_latency)
