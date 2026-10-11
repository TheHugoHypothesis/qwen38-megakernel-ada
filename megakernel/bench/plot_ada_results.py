#!/usr/bin/env python3
"""Plot recorded benchmarks; no inference runs or GPU configuration changes."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ada-report"
OUT.mkdir(exist_ok=True)
data = {
    name: json.loads((ROOT / f"results-ada-{name}" / "results.json").read_text())
    for name in ("validated", "stock-mtp", "sampling", "long")
}
summary = json.loads((ROOT / "ada_summary.json").read_text())
updated = all((ROOT / "optimization-analysis" / f"llama-mtp4-{case}" / "results.json").exists()
              for case in ("short", "sampling", "long"))
if updated:
    for case in ("short", "sampling", "long"):
        data[f"mtp4-{case}"] = json.loads((ROOT / "optimization-analysis" / f"llama-mtp4-{case}" / "results.json").read_text())


def llama(name):
    return [r["timings"]["predicted_per_second"] for r in data[name]["llama"]]


def mk(name, draft):
    return [r["tokens_per_second"] for r in data[name]["megakernel"] if r["draft"] == draft]


def number(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)).replace(".", ",")


short = [("llama.cpp · sem MTP", llama("validated")),
         ("llama.cpp · MTP 2", llama("stock-mtp"))]
if updated:
    short.append(("llama.cpp · MTP 4", llama("mtp4-short")))
short += [(f"Megakernel · MTP {d}" if d else "Megakernel · sem MTP", mk("validated", d))
          for d in range(5)]
cases = [("Greedy\n24 tokens de prompt", "stock-mtp", "validated"),
         ("Amostragem\n24 tokens de prompt", "sampling", "sampling"),
         ("Greedy\n1.365 tokens de prompt", "long", "long")]
if updated:
    cases = [("Greedy\n24 tokens de prompt", "mtp4-short", "validated"),
             ("Amostragem\n24 tokens de prompt", "mtp4-sampling", "sampling"),
             ("Greedy¹\n1.365 tokens de prompt", "mtp4-long", "long")]
comparison = []
agreement = []
teacher_agreement = []
for label, baseline, engine in cases:
    a, b = mean(llama(baseline)), mean(mk(engine, 4))
    comparison.append({"case": label.replace("\n", ": "), "llama_mtp_tps": a,
                       "megakernel_mtp4_tps": b, "gain_percent": 100 * (b / a - 1),
                       "llama_runs": len(llama(baseline)), "megakernel_runs": len(mk(engine, 4))})
for name in ("validated", "long"):
    ref = data[name]["llama"][0]["tokens"]
    rows = data[name]["megakernel"]
    assert all(len(row["ids"]) == len(ref) for row in rows)
    same = sum(sum(a == b for a, b in zip(row["ids"], ref)) for row in rows)
    total = sum(len(row["ids"]) for row in rows)
    agreement.append({"case": name, "equal_tokens": same, "compared_tokens": total,
                      "runs": len(rows), "prompt_tokens": len(data[name]["prompt_ids"]),
                      "percent": same / total * 100})
if updated:
    for case in ("short", "long"):
        score = data[f"mtp4-{case}"]["teacher_forced"]["stdout"]
        import re
        match = re.search(r"SCORE agree (\d+) / (\d+)", score)
        equal, total = map(int, match.groups())
        teacher_agreement.append({"case": case, "equal_tokens": equal, "compared_tokens": total,
                                  "percent": equal / total * 100})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREEN, RED = "#456b9a", "#16846d", "#bb4747"
fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"height_ratios": [1.15, 0.85]})
fig.subplots_adjust(left=0.19, right=0.97, top=0.86, bottom=0.14, wspace=0.28, hspace=0.47)
fig.suptitle("RTX 4060 Ti 16 GB · desempenho e correção atualizados", x=0.5, y=0.975,
             fontsize=21, fontweight="bold")
fig.text(0.5, 0.934, "Qwen3.8-27B GSQ-RCO IQ3_S · sm_89 · clocks padrão · KV Q8 · contexto alocado: 4.096",
         ha="center", color="#444444", fontsize=12)

ax = axes[0, 0]
vals = np.array([mean(v) for _, v in short])
err = np.array([[m - min(v) for m, (_, v) in zip(vals, short)],
                [max(v) - m for m, (_, v) in zip(vals, short)]])
y = np.arange(len(short))
ax.barh(y, vals, xerr=err, capsize=3, color=[BLUE] * (len(short) - 5) + [GREEN] * 5, height=0.65)
ax.set_yticks(y, [label for label, _ in short])
ax.invert_yaxis()
ax.set_xlim(0, max(vals) * 1.15)
ax.set_xlabel("Geração (tokens/s) · maior é melhor")
ax.set_title("Prompt curto greedy: efeito da profundidade MTP", loc="left", pad=12, fontsize=13)
ax.grid(axis="x", alpha=0.18)
ax.set_axisbelow(True)
for i, value in enumerate(vals):
    ax.text(value + 0.65, i, number(value), va="center", fontsize=10)

ax = axes[0, 1]
x = np.arange(3)
a = [r["llama_mtp_tps"] for r in comparison]
b = [r["megakernel_mtp4_tps"] for r in comparison]
ax.bar(x - 0.18, a, width=0.34, color=BLUE, label="llama.cpp · MTP 4" if updated else "llama.cpp · MTP 2")
ax.bar(x + 0.18, b, width=0.34, color=GREEN, label="Megakernel · MTP 4")
ax.set_xticks(x, [label for label, _, _ in cases])
ax.set_ylim(0, max(a + b) * 1.3)
ax.set_ylabel("Geração (tokens/s)")
ax.set_title("Comparação com llama.cpp usando MTP", loc="left", pad=12, fontsize=13)
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
for i, row in enumerate(comparison):
    for dx, value in ((-0.18, a[i]), (0.18, b[i])):
        ax.text(i + dx, value + 0.6, number(value), ha="center", fontsize=9)
    gain = row["gain_percent"]
    ax.text(i, max(a[i], b[i]) + 4.3, f"{gain:+.1f}%".replace(".", ","),
            ha="center", color=GREEN if gain >= 0 else RED, fontweight="bold", fontsize=13)

ax = axes[1, 0]
shown = teacher_agreement if updated else agreement
ax.barh([1, 0], [row["percent"] for row in shown], height=0.45,
        color=[GREEN, "#b47713"] if updated else GREEN)
ax.set_yticks([1, 0], ["Prompt curto", "Prompt de 1.365 tokens"] if updated else
             ["Prompt curto\n10 gerações", "Prompt de 1.365 tokens\n4 gerações"])
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_ylim(-0.65, 1.65)
ax.set_xlabel("Concordância dos argmax em teacher forcing (%)" if updated else "Tokens iguais à referência greedy (%)")
ax.set_title("Nova referência MTP 4: corretude ainda limitada" if updated else
             "Concordância observada: 2 prompts, 128 tokens/geração", loc="left", pad=12, fontsize=12)
ax.grid(axis="x", alpha=0.18)
ax.set_axisbelow(True)
for y, row in zip([1, 0], shown):
    ax.text(50, y, f"{number(row['percent'])}% · {row['equal_tokens']}/{row['compared_tokens']} tokens",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11)
ax.text(0, -0.36, "¹ Greedy longo: primeira divergência no índice 41 (base zero)." if updated else
        "Teacher forcing: 128/128 em cada prompt (também chunk 128).", fontsize=9, color="#444444")

ax = axes[1, 1]
ax.axis("off")
ax.set_title("O que a validação permite concluir", loc="left", pad=12, fontsize=13)
lines = [
    f"✓ {summary['quant_oracle_values']:,} valores dequantizados conferidos com GGML".replace(",", "."),
    f"✓ {summary['host_tests_passed']}/31 testes Python + teste do parser GGUF",
    f"✓ {summary['http_cuda_checks_passed']}/7 casos HTTP/CUDA; seed repetível",
    "✓ Memcheck: 0 erros em 3 ciclos de decode instrumentados",
    "",
    "Referência inicial: 14 gerações idênticas nos 2 prompts; não generaliza.",
    "Não medidos: perplexidade, acurácia em tarefas e avaliação ampla.",
    "Amostragem: IDs diferentes entre motores não medem perda de qualidade.",
    "DFlash2: não integrado ao megakernel; sem draft local para testar.",
]
for i, line in enumerate(lines):
    ax.text(0, 0.94 - i * 0.105, line, transform=ax.transAxes, fontsize=10,
            color=GREEN if i < 4 else "#444444", va="top")
fig.text(0.19, 0.075, "Médias de 2 repetições de 128 tokens (llama.cpp MTP greedy curto: 3). Barras de erro: mínimo–máximo, não intervalo de confiança.", fontsize=10, color="#444444")
fig.text(0.19, 0.049, "Throughput exclui carregamento e prefill. ¹ No caso longo, as continuações diferem; comparar também tempo de ciclo e aceitação.", fontsize=10, color="#444444")
fig.text(0.19, 0.023, "Fonte: bench/results-ada-* e bench/optimization-analysis/llama-mtp4-*. Clocks padrão; nenhuma estimativa de ganho DFlash2.", fontsize=9, color="#666666")
for extension in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"ganhos_e_correcao.{extension}", dpi=180, facecolor="white")
report = {"short_prompt_tps": [{"label": label, "runs": values, "mean": mean(values)} for label, values in short],
          "comparison": comparison, "initial_greedy_token_agreement": agreement,
          "updated_teacher_forcing": teacher_agreement,
          "limits": {"task_accuracy_measured": False, "perplexity_measured": False,
                     "dflash2_tested": False, "decode_excludes_prefill": True}}
(OUT / "chart_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved PNG, SVG, PDF and chart_data.json to", OUT)
