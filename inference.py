import json
import math
import os
from typing import Dict, Optional, Tuple

import fire
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


UAD_ALLOWED_OUTPUTS = {"normal", "abnormal"}


def resolve_quantization_mode(config):
    if config.get("load_in_4bit"):
        return "4bit"
    if config.get("load_in_8bit"):
        return "8bit"
    return None


def ensure_quantization_dependency(quantization_mode):
    if quantization_mode is None:
        return
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Quantized inference requested ({quantization_mode}) but bitsandbytes is not installed. "
            "Install bitsandbytes in the trafficllm environment first."
        ) from exc


def build_model_load_kwargs(config):
    quantization_mode = resolve_quantization_mode(config)
    kwargs = {"low_cpu_mem_usage": True}
    if quantization_mode == "4bit":
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    elif quantization_mode == "8bit":
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = "auto"
    return kwargs


def resolve_task_code(config, response):
    if response in config["tasks"]:
        return config["tasks"][response]
    task_aliases = config.get("task_aliases", {})
    if response in task_aliases:
        return task_aliases[response]
    raise KeyError(response)


def parse_uad_output(text: str) -> Optional[str]:
    cleaned = text.strip().lower()
    if cleaned in UAD_ALLOWED_OUTPUTS:
        return cleaned
    return None


def compute_window_abnormal_score(score_normal: float, score_abnormal: float) -> float:
    normal = math.exp(score_normal)
    abnormal = math.exp(score_abnormal)
    return abnormal / (normal + abnormal)


def build_uad_result(raw_output: str, score_normal: Optional[float] = None, score_abnormal: Optional[float] = None) -> Dict[str, object]:
    parsed_output = parse_uad_output(raw_output)
    if score_normal is not None and score_abnormal is not None:
        abnormal_score = compute_window_abnormal_score(score_normal=score_normal, score_abnormal=score_abnormal)
    elif parsed_output == "abnormal":
        abnormal_score = 1.0
    elif parsed_output == "normal":
        abnormal_score = 0.0
    else:
        abnormal_score = 0.5
    return {
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "is_valid_output": parsed_output is not None,
        "window_abnormal_score": abnormal_score,
    }


def load_model(model, ptuning_path, quantization_mode=None):
    if ptuning_path is not None:
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"),
            map_location="cpu",
        )
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        if quantization_mode is None:
            model = model.half().cuda()
        if hasattr(model.transformer, "prefix_encoder"):
            model.transformer.prefix_encoder.float()

    return model


def prompt_processing(prompt):
    instruction_text = prompt.split("<packet>")[0]
    traffic_data = "<packet>" + "<packet>".join(prompt.split("<packet>")[1:])
    return instruction_text, traffic_data


def preprompt(task, traffic_data):
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application category the encrypted beign or malicious traffic belongs to. The categories include BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the traffic belongs to. The categories include IRC, Neris, RBot, Virut, normal.\n",
        "WAD": "Classify the given HTTP request into benign and malicious categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an exception. Only output malicious or benign, no additional output is required. The given HTTP request is as follows:\n",
        "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an HTTP request, please output an exception. Only output abnormal or normal, no additional output is required. The given HTTP request is as follows:\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the encrypted VPN detection task to determine which behavior or application category the VPN encrypted traffic belongs to. The categories include aim, bittorrent, email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube.\n",
        "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application category the traffic belongs to under the Tor network. The categories include audio, browsing, chat, file, mail, p2p, video, voip.\n",
    }
    if task == "AAD":
        prompt = prepromt_set[task] + traffic_data.split("<packet>:")[1]
    else:
        prompt = prepromt_set[task] + traffic_data
    return prompt


def load_base_model(config):
    quantization_mode = resolve_quantization_mode(config)
    ensure_quantization_dependency(quantization_mode)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(
        config["model_path"],
        config=model_config,
        trust_remote_code=True,
        **build_model_load_kwargs(config),
    )
    return tokenizer, model, quantization_mode


def run_uad_mode(config: Dict[str, object], prompt: str) -> Dict[str, object]:
    tokenizer, model, quantization_mode = load_base_model(config)
    peft_key = config.get("uad_peft_key", "UAD")
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][peft_key])
    model_uad = load_model(model, ptuning_path, quantization_mode=quantization_mode)
    model_uad = model_uad.eval()
    response, _ = model_uad.chat(tokenizer, prompt, history=[])
    return build_uad_result(response)


def run_proxy_mode(config: Dict[str, object], prompt: str) -> Tuple[str, str]:
    tokenizer, model, quantization_mode = load_base_model(config)
    instruction_text, traffic_data = prompt_processing(prompt)

    ptuning_path = os.path.join(config["peft_path"], config["peft_set"]["NLP"])
    model_nlp = load_model(model, ptuning_path, quantization_mode=quantization_mode)
    model_nlp = model_nlp.eval()

    response, _ = model_nlp.chat(tokenizer, instruction_text, history=[])
    stage1 = response
    task = resolve_task_code(config, response)
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
    model_downstream = load_model(model, ptuning_path, quantization_mode=quantization_mode)
    model_downstream = model_downstream.eval()

    traffic_prompt = preprompt(task, traffic_data)
    response, _ = model_downstream.chat(tokenizer, traffic_prompt, history=[])
    stage2 = response
    return stage1, stage2


def main(config, prompt: str = None, **kwargs):
    with open(config, "r", encoding="utf-8") as fin:
        config_data = json.load(fin)

    task_name = config_data.get("task_name")
    if task_name == "UAD":
        result = run_uad_mode(config_data, prompt=prompt)
        print(json.dumps(result, ensure_ascii=False))
        return result

    stage1, stage2 = run_proxy_mode(config_data, prompt=prompt)
    print(stage1)
    print(stage2)
    return {"stage1": stage1, "stage2": stage2}


if __name__ == "__main__":
    fire.Fire(main)
