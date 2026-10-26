# How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models

Authors: *Parth Asawa\*, Alan Zhu\*, Abby O'Neill, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez*

*Equal contribution.

📜 Paper: [https://arxiv.org/pdf/2510.02453](https://arxiv.org/pdf/2510.02453)

## Setup

Run `scripts/setup_env.sh` to build both virtual environments: the root one
(`uv sync`, used by dataset construction and evaluation) and the training one
in `SkyRL/skyrl-train` (with vLLM, plus the packages the environments import).

### Black-box models: Claude on Vertex AI

All student / judge / summarizer calls go through `utils/llm.py`, which serves
them with Claude on Vertex AI via litellm. Model names in the code and datasets
are unchanged (e.g. `gpt-4o-mini`) and are mapped at call time:

| Original name | Claude model (override with) |
| --- | --- |
| `*-mini`, `*-nano` (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-mini`) | `claude-haiku-4-5` (`CLAUDE_SMALL_MODEL`) |
| other OpenAI names (`gpt-5`, `gpt-4.1`, `gpt-4o`, `o3`, ...) | `claude-sonnet-4-6` (`CLAUDE_LARGE_MODEL`) |
| `claude-*`, `anthropic/claude-*` | same Claude model, via Vertex |

Required environment:

```bash
export ANTHROPIC_VERTEX_PROJECT_ID=<gcp-project>   # or VERTEX_PROJECT
export CLOUD_ML_REGION=global                      # or VERTEX_LOCATION
export GOOGLE_APPLICATION_CREDENTIALS=<service-account-key.json>
```

`CLAUDE_MAX_TOKENS` (default 8192) sets `max_tokens`. The advisor itself is
still served locally by vLLM; calls with an `api_base` are not rerouted.
No `OPENAI_API_KEY` is needed. Set `WANDB_API_KEY` for wandb logging, or pass
`LOGGER=console`. Training scripts write checkpoints under `$CKPT_ROOT`
(default `$HOME`).

## Advisor Models

![image](assets/advisor_models.png)

### Overview

Customizing powerful, black-box models is a major challenge, with most practitioners typically limited to static prompting.

We propose a framework to train a small open-source “advisor” model to guide black-box models via feedback, optimizing the model to your specific environment, task, or users with RL.

We show that Advisor Models are highly effective for personalizing and adapting black-box models to specific environments. We additionally test the system in reasoning-intensive tasks, finding properties of the framework under which Advisor Models work best, and demonstrating the system robustness across models and environments.

### Example
![image](assets/example.png)

## 📜 License

**Advisor Models** is Apache 2.0 licensed, making it suitable for both academic and commercial use.

## 📧 Contact

Please feel free to reach out at pgasawa@berkeley.edu & aczhu@berkeley.edu!

## 📋 Citation

```text
@article{asawa2026trainadvisorsteeringblackbox,
  title={How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models},
  author={Parth Asawa and Alan Zhu and Abby O'Neill and Matei Zaharia and Alexandros G. Dimakis and Joseph E. Gonzalez},
  year={2026},
  journal={arXiv preprint arXiv:2510.02453},
} 
```