# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import math
from datetime import datetime
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify

from src.open_r1.trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import numbers

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    multi_reference_scanpath: bool = field(
        default=True,
        metadata={"help": "Average ScanMatch over all unique human references for the same training task."},
    )
    counterfactual_scanpath: bool = field(
        default=True,
        metadata={"help": "Use same-image, different-target human paths as counterfactual negative references."},
    )
    counterfactual_reward_weight: float = field(
        default=0.1,
        metadata={"help": "Weight of the margin-based counterfactual reward penalty."},
    )
    counterfactual_margin: float = field(
        default=0.0,
        metadata={"help": "Required positive-over-negative ScanMatch margin; also used by the human separability gate."},
    )
    counterfactual_human_gate: bool = field(
        default=True,
        metadata={"help": "Only use negative task pairs that are separable using held-out human references."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def process_points(bbox, reso):
    width, height = reso

    im_w = 512
    im_h = 384

    points = {key: [] for key in ["X", "Y", "T"]}
    for line in bbox:
        x = line[0] / width * im_w
        y = line[1] / height * im_h

        t = 0.5
        points["X"].append(x)
        points["Y"].append(y)
        points["T"].append(t)

    if len(points["X"]) < 3:
        for _ in range(3 - len(points["X"])):
            points["X"].append(1)
            points["Y"].append(1)
            points["T"].append(1)

    return points, im_w, im_h


def extract_points(answer_bbox):
    points = []
    for bbox in answer_bbox:
        if "Position" in bbox:
            answer = bbox["Position"]
            if len(answer) >= 2 and isinstance(answer[0], numbers.Number) and isinstance(answer[1], numbers.Number):
                points.append(answer[:2])

    return points


def process_scanpath(scanpath):
    scanpath = scanpath.replace("[ ", "[")
    scanpath = scanpath.replace(" ]", "]")
    scanpath = scanpath.replace(",", " ")
    scanpath = scanpath.replace("-", " ")
    return scanpath


def parse_scanpath_points(scanpath):
    scanpath = process_scanpath(scanpath.strip())
    points = re.findall(r'\[(\d+) \s*(\d+)\]', scanpath)
    return [[int(x), int(y)] for x, y in points]


def add_multi_reference_solutions(
    train_dataset,
    counterfactual=False,
    counterfactual_weight=0.1,
    counterfactual_margin=0.0,
    counterfactual_human_gate=True,
):
    """Attach positive and optional same-image negative references to training examples."""
    column_names = set(train_dataset.column_names)
    is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0

    if counterfactual_weight < 0 or counterfactual_margin < 0:
        raise ValueError("Counterfactual weight and margin must be non-negative.")

    if "reference_solutions" in column_names and not counterfactual:
        if is_main_process:
            print("Multi-reference ScanMatch: using existing reference_solutions column.")
        return train_dataset

    if not {"image_path", "solution"}.issubset(column_names):
        if is_main_process:
            print("Multi-reference ScanMatch: image_path/solution unavailable; using single-reference rewards.")
        return train_dataset

    if "target_id" in column_names:
        task_values = train_dataset["target_id"]
        grouping_description = "image_path + target_id"
    elif "problem" in column_names:
        # The released Stage 2 dataset has no target_id. Its problem contains the
        # target text and resolution, and is the task identity available there.
        task_values = train_dataset["problem"]
        grouping_description = "image_path + problem (target text)"
    else:
        if is_main_process:
            print("Multi-reference ScanMatch: task identity unavailable; using single-reference rewards.")
        return train_dataset

    image_paths = train_dataset["image_path"]
    solutions = train_dataset["solution"]
    task_keys = []
    references_by_task = {}
    tasks_by_image = {}

    for index, (image_path, task_value, solution) in enumerate(zip(image_paths, task_values, solutions)):
        valid_task = (
            image_path is not None
            and task_value is not None
            and isinstance(solution, str)
            and bool(solution.strip())
        )
        if not valid_task:
            task_keys.append(("single_row", index))
            continue

        task_key = ("task", image_path, str(task_value))
        task_keys.append(task_key)
        task_references = references_by_task.setdefault(task_key, {})
        normalized_solution = " ".join(solution.split())
        task_references.setdefault(normalized_solution, solution.strip())
        tasks_by_image.setdefault(image_path, {})[task_key] = None

    negative_references_by_task = {}
    if counterfactual:
        for task_key, positive_references in references_by_task.items():
            image_path = task_key[1]
            negative_references = {}
            for negative_task_key in tasks_by_image[image_path]:
                if negative_task_key == task_key:
                    continue
                for normalized_solution, negative_solution in references_by_task[negative_task_key].items():
                    # An identical path shared by both targets is not evidence
                    # against the current target and must not become a negative.
                    if normalized_solution not in positive_references:
                        negative_references.setdefault(normalized_solution, negative_solution)
            negative_references_by_task[task_key] = list(negative_references.values())

    reference_solutions = []
    negative_reference_solutions = []
    for task_key, solution in zip(task_keys, solutions):
        references = list(references_by_task.get(task_key, {}).values())
        if not references and isinstance(solution, str) and solution.strip():
            references = [solution.strip()]
        reference_solutions.append(references)
        negative_reference_solutions.append(negative_references_by_task.get(task_key, []))

    task_sizes = [len(references) for key, references in references_by_task.items() if key[0] == "task"]
    num_tasks = len(task_sizes)
    multi_reference_tasks = sum(size > 1 for size in task_sizes)
    multi_reference_rows = sum(len(references) > 1 for references in reference_solutions)
    counterfactual_rows = sum(bool(references) for references in negative_reference_solutions)

    if is_main_process:
        task_percentage = 100.0 * multi_reference_tasks / num_tasks if num_tasks else 0.0
        row_percentage = 100.0 * multi_reference_rows / len(train_dataset) if len(train_dataset) else 0.0
        print(
            "Multi-reference ScanMatch: "
            f"grouping={grouping_description}, tasks={num_tasks}, "
            f"multi_reference_tasks={multi_reference_tasks} ({task_percentage:.2f}%), "
            f"rows_with_multiple_references={multi_reference_rows}/{len(train_dataset)} "
            f"({row_percentage:.2f}%), max_references={max(task_sizes, default=1)}."
        )
        if counterfactual:
            counterfactual_percentage = (
                100.0 * counterfactual_rows / len(train_dataset) if len(train_dataset) else 0.0
            )
            print(
                "Counterfactual ScanMatch: "
                f"candidate_rows={counterfactual_rows}/{len(train_dataset)} "
                f"({counterfactual_percentage:.2f}%), weight={counterfactual_weight}, "
                f"margin={counterfactual_margin}, human_gate={counterfactual_human_gate}. "
                "Human separability is evaluated lazily and cached per task pair."
            )

    result = train_dataset
    if "reference_solutions" not in column_names:
        result = result.add_column("reference_solutions", reference_solutions)
    elif is_main_process:
        print("Multi-reference ScanMatch: using existing reference_solutions column.")

    if counterfactual:
        result = result.add_column("negative_reference_solutions", negative_reference_solutions)
        result = result.add_column(
            "counterfactual_reward_weight", [float(counterfactual_weight)] * len(result)
        )
        result = result.add_column("counterfactual_margin", [float(counterfactual_margin)] * len(result))
        result = result.add_column(
            "counterfactual_human_gate", [bool(counterfactual_human_gate)] * len(result)
        )

    return result


def scanmatch_scores_for_references(student_points, references, resolution, scanpath_evaluator):
    scores = []
    reference_bboxes = []
    for reference in references:
        if not isinstance(reference, str) or not reference.strip():
            continue
        try:
            reference_bbox = parse_scanpath_points(reference)
            if not reference_bbox:
                continue
            ground_truth_points, im_w, im_h = process_points(reference_bbox, resolution)
            score = float(scanpath_evaluator.ScanMatchRL(student_points, ground_truth_points, im_w, im_h))
            if not math.isfinite(score):
                continue
            score = min(1.0, max(0.0, score))
        except Exception:
            continue
        scores.append(score)
        reference_bboxes.append(reference_bbox)
    return scores, reference_bboxes


def mean_scanmatch_between_solutions(student_solution, references, resolution, scanpath_evaluator):
    try:
        student_bbox = parse_scanpath_points(student_solution)
        if not student_bbox:
            return None
        student_points, _, _ = process_points(student_bbox, resolution)
    except Exception:
        return None
    scores, _ = scanmatch_scores_for_references(
        student_points, references, resolution, scanpath_evaluator
    )
    return sum(scores) / len(scores) if scores else None


@lru_cache(maxsize=4096)
def counterfactual_human_reliability(
    scanpath_evaluator,
    positive_references,
    negative_references,
    resolution,
):
    """Estimate same-image task separability using held-out human paths only."""
    margins = []

    def add_human_margins(own_references, other_references):
        if len(own_references) < 2:
            return
        for index, human_solution in enumerate(own_references):
            held_out_references = own_references[:index] + own_references[index + 1:]
            own_score = mean_scanmatch_between_solutions(
                human_solution, held_out_references, resolution, scanpath_evaluator
            )
            other_score = mean_scanmatch_between_solutions(
                human_solution, other_references, resolution, scanpath_evaluator
            )
            if own_score is not None and other_score is not None:
                margins.append(own_score - other_score)

    # If one side has only one human path, the other side can still provide a
    # leave-one-out estimate. If both are singletons, the pair is unreliable.
    add_human_margins(positive_references, negative_references)
    add_human_margins(negative_references, positive_references)
    return sum(margins) / len(margins) if margins else None


def expand_reward_batch_parameter(value, batch_size, name):
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"{name} must have the same batch size as completions.")
        return value
    return [value] * batch_size


def accuracy_reward_scanpath(
    completions,
    solution,
    scanpath_evaluator,
    resolution,
    reference_solutions=None,
    negative_reference_solutions=None,
    counterfactual_reward_weight=0.0,
    counterfactual_margin=0.0,
    counterfactual_human_gate=True,
    **kwargs,
):
    """Multi-reference ScanMatch with an optional counterfactual margin penalty."""
    contents = [completion[0]["content"] for completion in completions]
    batch_size = len(contents)
    if reference_solutions is None:
        reference_solutions = [None] * batch_size
    if len(reference_solutions) != batch_size:
        raise ValueError("reference_solutions must have the same batch size as completions.")
    if negative_reference_solutions is None:
        negative_reference_solutions = [None] * batch_size
    negative_reference_solutions = expand_reward_batch_parameter(
        negative_reference_solutions, batch_size, "negative_reference_solutions"
    )
    counterfactual_reward_weight = expand_reward_batch_parameter(
        counterfactual_reward_weight, batch_size, "counterfactual_reward_weight"
    )
    counterfactual_margin = expand_reward_batch_parameter(
        counterfactual_margin, batch_size, "counterfactual_margin"
    )
    counterfactual_human_gate = expand_reward_batch_parameter(
        counterfactual_human_gate, batch_size, "counterfactual_human_gate"
    )

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for batch_index, (content, sol, reso, references) in enumerate(
        zip(contents, solution, resolution, reference_solutions)
    ):
        positive_reward = 0.0
        final_reward = 0.0
        student_answer_bbox = []
        ground_truth_bbox = []
        ground_truth_reference_bboxes = []
        reference_rewards = []
        negative_rewards = []
        negative_reward = None
        human_reliability = None
        counterfactual_penalty = 0.0
        counterfactual_active = False
        negative_list = negative_reference_solutions[batch_index]
        weight = float(counterfactual_reward_weight[batch_index])
        margin = float(counterfactual_margin[batch_index])
        use_human_gate = bool(counterfactual_human_gate[batch_index])

        if references is None:
            reference_list = [sol]
        elif isinstance(references, str):
            reference_list = [references]
        else:
            reference_list = list(references)
            if not reference_list:
                reference_list = [sol]

        if negative_list is None:
            negative_list = []
        elif isinstance(negative_list, str):
            negative_list = [negative_list]
        else:
            negative_list = list(negative_list)

        try:
            content_match = re.search(r'<answer>(.*?)</answer>', content.strip())
            content_answer = content_match.group(1).strip() if content_match else content.strip()
            student_answer_bbox = parse_scanpath_points(content_answer)
            student_points, _, _ = process_points(student_answer_bbox, reso) if student_answer_bbox else (None, None, None)
        except Exception:
            student_answer_bbox = []
            student_points = None

        if student_points is not None:
            reference_rewards, ground_truth_reference_bboxes = scanmatch_scores_for_references(
                student_points, reference_list, reso, scanpath_evaluator
            )
            if ground_truth_reference_bboxes:
                ground_truth_bbox = ground_truth_reference_bboxes[0]

        if reference_rewards:
            positive_reward = sum(reference_rewards) / len(reference_rewards)
            final_reward = positive_reward

        if student_points is not None and negative_list and weight > 0:
            if use_human_gate:
                human_reliability = counterfactual_human_reliability(
                    scanpath_evaluator,
                    tuple(reference_list),
                    tuple(negative_list),
                    tuple(reso),
                )
                counterfactual_active = human_reliability is not None and human_reliability > margin
            else:
                counterfactual_active = True

            if counterfactual_active:
                negative_rewards, _ = scanmatch_scores_for_references(
                    student_points, negative_list, reso, scanpath_evaluator
                )
                if negative_rewards:
                    negative_reward = sum(negative_rewards) / len(negative_rewards)
                    counterfactual_penalty = max(0.0, negative_reward + margin - positive_reward)
                    final_reward = positive_reward - weight * counterfactual_penalty

        rewards.append(final_reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} Accuracy reward of ScanMatch: {final_reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"sol: {sol}\n")
                    f.write(f"reference_solutions: {reference_list}\n")
                    f.write(f"negative_reference_solutions: {negative_list}\n")
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    f.write(f"ground_truth_reference_bboxes: {ground_truth_reference_bboxes}\n")
                    f.write(f"reference_scanmatch_results: {reference_rewards}\n")
                    f.write(f"negative_scanmatch_results: {negative_rewards}\n")
                    f.write(f"num_references: {len(reference_rewards)}\n")
                    f.write(f"positive_scanmatch_mean: {positive_reward}\n")
                    f.write(f"negative_scanmatch_mean: {negative_reward}\n")
                    f.write(f"counterfactual_human_reliability: {human_reliability}\n")
                    f.write(f"counterfactual_active: {counterfactual_active}\n")
                    f.write(f"counterfactual_weight: {weight}\n")
                    f.write(f"counterfactual_margin: {margin}\n")
                    f.write(f"counterfactual_penalty: {counterfactual_penalty}\n")
                    f.write(f"scanmatch_results: {final_reward}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

###  reward registry three parts
reward_funcs_registry = {
    "accuracy_scanpath": accuracy_reward_scanpath,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_scanpath']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    train_split = script_args.dataset_train_split
    if script_args.counterfactual_scanpath and not script_args.multi_reference_scanpath:
        raise ValueError("counterfactual_scanpath requires multi_reference_scanpath=true.")
    if script_args.multi_reference_scanpath:
        # Build references from the training split only; evaluation/test trajectories
        # must never become training rewards.
        dataset[train_split] = add_multi_reference_solutions(
            dataset[train_split],
            counterfactual=script_args.counterfactual_scanpath,
            counterfactual_weight=script_args.counterfactual_reward_weight,
            counterfactual_margin=script_args.counterfactual_margin,
            counterfactual_human_gate=script_args.counterfactual_human_gate,
        )
    else:
        print("Multi-reference ScanMatch: disabled; using single-reference rewards.")

    assert training_args.use_vllm is False
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    print("--- Debugging Training Arguments ---")
    print(training_args)
    print("------------------------------------")

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
