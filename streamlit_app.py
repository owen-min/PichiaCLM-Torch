from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict

import requests
import streamlit as st

from Model_PichiaCLM.core.analysis import analyze_cds, load_training_codon_reference
from Model_PichiaCLM.core.biology import normalize_dna
from Model_PichiaCLM.core.config import DEFAULT_WEIGHTS_PATH
from Model_PichiaCLM.core.fasta import FastaRecord, format_fasta, parse_fasta
from Model_PichiaCLM.core.fusion import compare_signal_fusion
from Model_PichiaCLM.core.postprocess import conservative_postprocess
from Model_PichiaCLM.core.predictor import PichiaCLMPredictor


DEFAULT_SEQUENCE = "MSTNPKPQR"
DIRECT_MODE_LABEL = "直接加载模型"
API_MODE_LABEL = "调用 FastAPI"


@st.cache_resource(show_spinner="正在加载 PichiaCLM 模型...")
def load_predictor(weights_path: str, device: str | None) -> PichiaCLMPredictor:
    return PichiaCLMPredictor(weights_path=weights_path, device=device or None)


def parse_text_list(raw_text: str) -> list[str]:
    return [item.strip() for item in raw_text.replace(",", "\n").splitlines() if item.strip()]


def json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def gc_status_label(status: str) -> str:
    return {"ok": "正常", "low": "偏低", "high": "偏高"}.get(status, status)


def quality_counts(analysis: dict[str, object]) -> tuple[int, int]:
    critical = 0
    if not analysis["valid_dna"]:
        critical += 1
    if not analysis["length_multiple_of_three"]:
        critical += 1
    if analysis["translation_matches_input"] is False:
        critical += 1
    critical += len(analysis["internal_stop_codons"])

    warnings = len(analysis["sequence_warnings"])
    warnings += 1 if analysis["gc_status"] != "ok" else 0
    warnings += len(analysis["local_gc_outliers"])
    warnings += len(analysis["restriction_sites"])
    warnings += len(analysis["motif_hits"])
    warnings += len(analysis["rare_codon_runs"])
    warnings += len(analysis["homopolymers"])
    warnings += len(analysis["tandem_repeats"])
    warnings += len(analysis["repeated_kmers"])
    return critical, warnings


def quality_status(analysis: dict[str, object]) -> str:
    critical, warnings = quality_counts(analysis)
    if critical:
        return "失败"
    if warnings:
        return "警告"
    return "通过"


def translation_status(analysis: dict[str, object]) -> str:
    if analysis["translation_matches_input"] is None:
        return "未提供参考 AA"
    return "通过" if analysis["translation_matches_input"] else "未通过"


def render_quality_overview(analysis: dict[str, object]) -> None:
    critical, warnings = quality_counts(analysis)
    status = quality_status(analysis)
    if status == "通过":
        st.success("总体结论: 通过当前质量检查")
    elif status == "警告":
        st.warning("总体结论: 存在需要复核的风险项")
    else:
        st.error("总体结论: 存在基础正确性问题")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("结论", status)
    col_b.metric("基础问题", critical)
    col_c.metric("风险警告", warnings)


def dataframe_or_success(title: str, rows: list[dict[str, object]], empty_text: str = "未发现") -> None:
    if rows:
        st.warning(f"{title}: 发现 {len(rows)} 处")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.success(f"{title}: {empty_text}")


def predict_direct(
    amino_acids: str,
    allow_unknown: bool,
    weights_path: str,
    device: str | None,
    motifs: list[str],
    custom_sites: list[str],
    do_postprocess: bool,
) -> dict[str, object]:
    predictor = load_predictor(weights_path, device)
    result = predictor.predict(amino_acids, allow_unknown=allow_unknown)
    payload = asdict(result)
    payload["analysis"] = asdict(
        analyze_cds(
            result.cds,
            amino_acids=result.amino_acids,
            motifs=motifs,
            custom_restriction_sites=custom_sites,
        )
    )
    if do_postprocess:
        training_reference, _ = load_training_codon_reference()
        payload["postprocess"] = asdict(
            conservative_postprocess(
                result.cds,
                result.amino_acids,
                reference_fractions=training_reference,
                forbidden_motifs=motifs,
                custom_restriction_sites=custom_sites,
            )
        )
    return payload


def predict_via_api(
    api_url: str,
    amino_acids: str,
    allow_unknown: bool,
    motifs: list[str],
    custom_sites: list[str],
    do_postprocess: bool,
) -> dict[str, object]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={
            "amino_acids": amino_acids,
            "allow_unknown": allow_unknown,
            "include_analysis": True,
            "unwanted_motifs": motifs,
            "custom_restriction_sites": custom_sites,
            "postprocess": do_postprocess,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def analyze_external_cds_direct(
    cds: str,
    expected_amino_acids: str | None,
    motifs: list[str],
    custom_sites: list[str],
) -> dict[str, object]:
    analysis = analyze_cds(
        cds,
        amino_acids=expected_amino_acids,
        motifs=motifs,
        custom_restriction_sites=custom_sites,
    )
    return {
        "cds": normalize_dna(cds),
        "expected_amino_acids": expected_amino_acids,
        "translated_amino_acids": analysis.translated_amino_acids,
        "analysis": asdict(analysis),
    }


def analyze_external_cds_via_api(
    api_url: str,
    cds: str,
    expected_amino_acids: str | None,
    motifs: list[str],
    custom_sites: list[str],
) -> dict[str, object]:
    response = requests.post(
        f"{api_url.rstrip('/')}/analyze_cds",
        json={
            "cds": cds,
            "expected_amino_acids": expected_amino_acids,
            "unwanted_motifs": motifs,
            "custom_restriction_sites": custom_sites,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def run_cds_analysis(
    mode: str,
    cds: str,
    expected_amino_acids: str | None,
    api_url: str,
    motifs: list[str],
    custom_sites: list[str],
) -> dict[str, object]:
    if mode == DIRECT_MODE_LABEL:
        return analyze_external_cds_direct(cds, expected_amino_acids, motifs, custom_sites)
    return analyze_external_cds_via_api(api_url, cds, expected_amino_acids, motifs, custom_sites)


def run_prediction(
    mode: str,
    amino_acids: str,
    allow_unknown: bool,
    weights_path: str,
    device: str | None,
    api_url: str,
    motifs: list[str],
    custom_sites: list[str],
    do_postprocess: bool,
) -> dict[str, object]:
    if mode == DIRECT_MODE_LABEL:
        return predict_direct(amino_acids, allow_unknown, weights_path, device, motifs, custom_sites, do_postprocess)
    return predict_via_api(api_url, amino_acids, allow_unknown, motifs, custom_sites, do_postprocess)


def render_cds_analysis_result(result: dict[str, object], title: str = "CDS 质检结果", key_prefix: str = "cds_qc") -> None:
    analysis = result["analysis"]
    st.subheader(title)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("CDS 长度", analysis["cds_length"])
    col_b.metric("密码子数量", analysis["codon_count"])
    col_c.metric("翻译一致性", translation_status(analysis))

    st.text_area("待质检 CDS", value=result["cds"], height=120, key=f"{key_prefix}_cds_text")
    st.text_area("翻译得到的 AA", value=result["translated_amino_acids"], height=100, key=f"{key_prefix}_translated_aa")

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("GC 含量", f"{analysis['gc_percent']}%", gc_status_label(analysis["gc_status"]))
    metric_b.metric("局部 GC 警告", len(analysis["local_gc_outliers"]))
    metric_c.metric("CAI 训练数据", analysis["cai"]["training"])
    metric_d.metric("CAI 公开表", analysis["cai"]["public"])

    check_a, check_b, check_c = st.columns(3)
    check_a.metric("酶切位点", len(analysis["restriction_sites"]))
    check_b.metric("Motif 命中", len(analysis["motif_hits"]))
    check_c.metric("非法碱基", len(analysis["invalid_bases"]))

    render_quality_overview(analysis)
    render_quality_report(analysis)
    with st.expander("密码子使用对比"):
        used_rows = [
            {
                "密码子": row["codon"],
                "氨基酸": row["amino_acid"],
                "出现次数": row["count"],
                "本序列比例": row["sequence_fraction"],
                "训练数据比例": row["training_fraction"],
                "公开表比例": row["public_fraction"],
            }
            for row in analysis["codon_usage"]
            if row["count"] > 0
        ]
        st.dataframe(used_rows, width="stretch", hide_index=True)

    st.download_button(
        "下载 CDS 质检报告 JSON",
        data=json_dumps(result),
        file_name="pichiaclm_cds_qc.json",
        mime="application/json",
        key=f"{key_prefix}_json_download",
    )


def render_prediction_result(result: dict[str, object], title: str = "预测结果", key_prefix: str = "result") -> None:
    analysis = result["analysis"]
    st.subheader(title)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("氨基酸长度", len(result["amino_acids"]))
    col_b.metric("CDS 长度", len(result["cds"]))
    col_c.metric("运行设备", result["device"])

    st.text_area("优化后的 CDS", value=result["cds"], height=120, key=f"{key_prefix}_cds_text")
    st.download_button(
        "下载 CDS FASTA",
        data=f">PichiaCLM_prediction\n{result['cds']}\n",
        file_name="pichiaclm_prediction.fasta",
        mime="text/plain",
        key=f"{key_prefix}_cds_download",
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("GC 含量", f"{analysis['gc_percent']}%", gc_status_label(analysis["gc_status"]))
    metric_b.metric("局部 GC 警告", len(analysis["local_gc_outliers"]))
    metric_c.metric("CAI 训练数据", analysis["cai"]["training"])
    metric_d.metric("CAI 公开表", analysis["cai"]["public"])

    check_a, check_b, check_c, check_d = st.columns(4)
    check_a.metric("翻译一致性", translation_status(analysis))
    check_b.metric("酶切位点", len(analysis["restriction_sites"]))
    check_c.metric("Motif 命中", len(analysis["motif_hits"]))
    check_d.metric("连续稀有密码子", len(analysis["rare_codon_runs"]))

    st.caption(
        "GC 阈值: 全局 35%-65%；局部 30 bp 窗口 25%-75%。"
        "公开 CAI 参考表: Kazusa Pichia pastoris taxon 4922。"
    )
    render_quality_overview(analysis)
    render_quality_report(analysis)
    render_postprocess(result, key_prefix=key_prefix)

    with st.expander("密码子使用对比"):
        used_rows = [
            {
                "密码子": row["codon"],
                "氨基酸": row["amino_acid"],
                "出现次数": row["count"],
                "本序列比例": row["sequence_fraction"],
                "训练数据比例": row["training_fraction"],
                "公开表比例": row["public_fraction"],
            }
            for row in analysis["codon_usage"]
            if row["count"] > 0
        ]
        st.dataframe(used_rows, width="stretch", hide_index=True)

    st.download_button(
        "下载分析报告 JSON",
        data=json_dumps(result),
        file_name="pichiaclm_analysis.json",
        mime="application/json",
        key=f"{key_prefix}_json_download",
    )


def render_quality_report(analysis: dict[str, object]) -> None:
    st.subheader("质量警告")
    with st.expander("基础正确性检查", expanded=True):
        for warning in analysis["sequence_warnings"]:
            st.warning(warning)
        st.success("DNA 字母检查: 通过" if analysis["valid_dna"] else "DNA 字母检查: 未通过")
        st.success("阅读框检查: 通过" if analysis["length_multiple_of_three"] else "阅读框检查: 未通过")
        if analysis["translation_matches_input"] is None:
            st.info("翻译一致性: 未提供参考 AA")
        else:
            st.success("翻译一致性: 通过" if analysis["translation_matches_input"] else "翻译一致性: 未通过")
        dataframe_or_success(
            "内部终止密码子",
            [{"密码子编号": codon_number} for codon_number in analysis["internal_stop_codons"]],
        )

    with st.expander("GC 含量"):
        st.write(f"全局 GC 含量: {analysis['gc_percent']}% ({gc_status_label(analysis['gc_status'])})")
        dataframe_or_success(
            "局部 GC 异常区域",
            [
                {"起始 bp": row["start"], "结束 bp": row["end"], "GC (%)": row["gc_percent"]}
                for row in analysis["local_gc_outliers"][:50]
            ],
        )

    with st.expander("酶切位点与 motif"):
        dataframe_or_success(
            "限制性酶切位点",
            [
                {
                    "酶名": row["name"],
                    "识别序列": row["sequence"],
                    "起始 bp": row["start"],
                    "结束 bp": row["end"],
                }
                for row in analysis["restriction_sites"]
            ],
        )
        dataframe_or_success(
            "不想要的 motif",
            [
                {"Motif 序列": row["motif"], "起始 bp": row["start"], "结束 bp": row["end"]}
                for row in analysis["motif_hits"]
            ],
        )

    with st.expander("密码子与重复序列"):
        dataframe_or_success(
            "连续稀有密码子",
            [
                {
                    "参考表": "训练数据" if row["reference"] == "training" else "公开表",
                    "起始密码子": row["start_codon"],
                    "结束密码子": row["end_codon"],
                    "密码子": " ".join(row["codons"]),
                }
                for row in analysis["rare_codon_runs"][:50]
            ],
        )
        dataframe_or_success(
            "长同聚碱基",
            [
                {"碱基": row["base"], "起始 bp": row["start"], "结束 bp": row["end"], "长度": row["length"]}
                for row in analysis["homopolymers"][:50]
            ],
        )
        dataframe_or_success(
            "串联重复",
            [
                {
                    "重复单元": row["sequence"],
                    "起始 bp": row["start"],
                    "结束 bp": row["end"],
                    "重复次数": row["copies"],
                }
                for row in analysis["tandem_repeats"][:50]
            ],
        )
        dataframe_or_success(
            "重复 12 bp 片段",
            [
                {
                    "重复片段": row["sequence"],
                    "出现次数": row["count"],
                    "前几个位置": ", ".join(str(position) for position in row["positions"]),
                }
                for row in analysis["repeated_kmers"][:50]
            ],
        )


def render_postprocess(result: dict[str, object], key_prefix: str) -> None:
    postprocess = result.get("postprocess")
    if not postprocess:
        return
    st.subheader("后处理建议")
    st.metric("同义替换次数", len(postprocess["replacements"]))
    st.success("翻译保持一致" if postprocess["translation_preserved"] else "翻译未保持一致")
    if postprocess["optimized_cds"] != postprocess["original_cds"]:
        st.text_area("后处理后的 CDS", value=postprocess["optimized_cds"], height=120, key=f"{key_prefix}_post_cds")
    dataframe_or_success(
        "替换记录",
        [
            {
                "密码子编号": row["codon_number"],
                "位置": f"{row['start']}-{row['end']}",
                "氨基酸": row["amino_acid"],
                "原密码子": row["old_codon"],
                "新密码子": row["new_codon"],
                "原因": row["reason"],
            }
            for row in postprocess["replacements"]
        ],
    )
    dataframe_or_success("仍未解决的问题", [{"问题": item} for item in postprocess["remaining_issues"]])


def records_to_csv(rows: list[dict[str, object]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def sidebar_settings() -> dict[str, object]:
    with st.sidebar:
        mode = st.radio("预测模式", [DIRECT_MODE_LABEL, API_MODE_LABEL], index=0)
        if mode == DIRECT_MODE_LABEL:
            weights_path = st.text_input("模型权重路径", value=str(DEFAULT_WEIGHTS_PATH))
            device = st.selectbox("运行设备", ["auto", "cpu", "cuda"], index=0)
            api_url = ""
        else:
            api_url = st.text_input("API 服务地址", value="http://127.0.0.1:8000")
            weights_path = str(DEFAULT_WEIGHTS_PATH)
            device = "auto"
        with st.expander("高级预测设置"):
            allow_unknown = st.checkbox("允许模糊氨基酸", value=False)
            do_postprocess = st.checkbox("启用保守后处理", value=False)
        with st.expander("高级质检设置"):
            motifs = parse_text_list(st.text_area("不想要的 motif", value="", height=90))
            custom_sites = parse_text_list(st.text_area("自定义酶切位点", value="", height=90))
    return {
        "mode": mode,
        "allow_unknown": allow_unknown,
        "do_postprocess": do_postprocess,
        "motifs": motifs,
        "custom_sites": custom_sites,
        "weights_path": weights_path,
        "device": None if device == "auto" else device,
        "api_url": api_url,
    }


def render_single_tab(settings: dict[str, object]) -> None:
    amino_acids = st.text_area("氨基酸序列", value=DEFAULT_SEQUENCE, height=160)
    if st.button("开始预测", type="primary", key="single_predict"):
        try:
            result = run_prediction(amino_acids=amino_acids, **settings)
        except Exception as exc:
            st.error(str(exc))
            return
        render_prediction_result(result)


def render_batch_tab(settings: dict[str, object]) -> None:
    uploaded = st.file_uploader("上传 AA FASTA 文件", type=["fa", "fasta", "faa", "txt"])
    raw_fasta = st.text_area("或粘贴 AA FASTA", value="", height=180)
    if st.button("批量预测", type="primary", key="batch_predict"):
        try:
            text = uploaded.read().decode("utf-8") if uploaded is not None else raw_fasta
            records = parse_fasta(text)
            results = []
            cds_records = []
            summary_rows = []
            for record in records:
                result = run_prediction(amino_acids=record.sequence, **settings)
                result["id"] = record.id
                result["description"] = record.description
                results.append(result)
                output_cds = result.get("postprocess", {}).get("optimized_cds", result["cds"])
                cds_records.append(FastaRecord(id=record.id, description="PichiaCLM optimized CDS", sequence=output_cds))
                analysis = result["analysis"]
                critical, warnings = quality_counts(analysis)
                summary_rows.append(
                    {
                        "ID": record.id,
                        "结论": quality_status(analysis),
                        "基础问题": critical,
                        "风险警告": warnings,
                        "AA 长度": len(result["amino_acids"]),
                        "CDS 长度": len(output_cds),
                        "GC%": analysis["gc_percent"],
                        "CAI 训练数据": analysis["cai"]["training"],
                        "CAI 公开表": analysis["cai"]["public"],
                        "翻译一致": analysis["translation_matches_input"],
                        "酶切位点": len(analysis["restriction_sites"]),
                        "Motif 命中": len(analysis["motif_hits"]),
                        "局部 GC 警告": len(analysis["local_gc_outliers"]),
                    }
                )
        except Exception as exc:
            st.error(str(exc))
            return

        st.subheader("批量结果")
        summary_rows.sort(key=lambda row: (row["基础问题"], row["风险警告"]), reverse=True)
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        st.download_button("下载 CDS FASTA", data=format_fasta(cds_records), file_name="pichiaclm_batch_cds.fasta")
        st.download_button("下载结果表格 CSV", data=records_to_csv(summary_rows), file_name="pichiaclm_batch_report.csv")
        with st.expander("逐条详细结果"):
            for result in results:
                render_prediction_result(result, title=f"{result['id']} 详细结果", key_prefix=f"batch_{result['id']}")


def render_fusion_tab(settings: dict[str, object]) -> None:
    signal = st.text_area("信号肽 AA", value="", height=100)
    mature = st.text_area("成熟蛋白 AA", value=DEFAULT_SEQUENCE, height=140)
    if st.button("比较整体优化与分段优化", type="primary", key="fusion_predict"):
        try:
            if settings["mode"] != DIRECT_MODE_LABEL:
                st.warning("信号肽拼接对比需要直接加载模型模式。")
                return
            predictor = load_predictor(settings["weights_path"], settings["device"])
            comparison = compare_signal_fusion(
                predictor,
                signal_peptide=signal,
                mature_protein=mature,
                allow_unknown=settings["allow_unknown"],
            )
            payload = asdict(comparison)
        except Exception as exc:
            st.error(str(exc))
            return

        st.subheader("信号肽拼接对比")
        st.metric("两种 CDS 是否完全一致", "是" if payload["cds_are_identical"] else "否")
        for label, key in [("整体优化", "whole_sequence"), ("分段优化", "segmented")]:
            item = payload[key]
            st.markdown(f"### {label}")
            render_prediction_result(
                {
                    **item["prediction"],
                    "analysis": item["analysis"],
                },
                title=f"{label}结果",
                key_prefix=f"fusion_{key}",
            )
            window = item["cleavage_window"]
            st.write(
                f"切割位点附近窗口: AA {window['amino_acid_start']}-{window['amino_acid_end']} / "
                f"CDS {window['cds_start']}-{window['cds_end']}"
            )
            st.code(window["amino_acids"], language="text")
            st.code(window["cds"], language="text")


def render_external_cds_tab(settings: dict[str, object]) -> None:
    st.subheader("二次优化 CDS 质检")
    st.caption("用于检查在外部软件中二次优化后的 CDS；这里不会重新调用模型预测。")

    single_cds = st.text_area("粘贴二次优化后的 CDS", value="", height=160, key="external_single_cds")
    expected_aa = st.text_area("可选：期望翻译得到的 AA", value="", height=120, key="external_expected_aa")
    if st.button("分析这条 CDS", type="primary", key="external_single_analyze"):
        try:
            result = run_cds_analysis(
                mode=settings["mode"],
                cds=single_cds,
                expected_amino_acids=expected_aa or None,
                api_url=settings["api_url"],
                motifs=settings["motifs"],
                custom_sites=settings["custom_sites"],
            )
        except Exception as exc:
            st.error(str(exc))
            return
        render_cds_analysis_result(result, key_prefix="external_single")

    st.divider()
    st.subheader("批量 CDS FASTA 质检")
    uploaded = st.file_uploader("上传 CDS FASTA 文件", type=["fa", "fasta", "fna", "txt"], key="external_cds_upload")
    raw_fasta = st.text_area("或粘贴 CDS FASTA", value="", height=180, key="external_cds_fasta")
    batch_expected_aa = st.text_area("可选：所有 CDS 共用的期望 AA", value="", height=100, key="external_batch_expected")

    if st.button("批量质检 CDS", type="primary", key="external_batch_analyze"):
        try:
            text = uploaded.read().decode("utf-8") if uploaded is not None else raw_fasta
            records = parse_fasta(text)
            results = []
            summary_rows = []
            for record in records:
                result = run_cds_analysis(
                    mode=settings["mode"],
                    cds=record.sequence,
                    expected_amino_acids=batch_expected_aa or None,
                    api_url=settings["api_url"],
                    motifs=settings["motifs"],
                    custom_sites=settings["custom_sites"],
                )
                result["id"] = record.id
                result["description"] = record.description
                results.append(result)
                analysis = result["analysis"]
                critical, warnings = quality_counts(analysis)
                summary_rows.append(
                    {
                        "ID": record.id,
                        "结论": quality_status(analysis),
                        "基础问题": critical,
                        "风险警告": warnings,
                        "CDS 长度": analysis["cds_length"],
                        "密码子数量": analysis["codon_count"],
                        "GC%": analysis["gc_percent"],
                        "CAI 训练数据": analysis["cai"]["training"],
                        "CAI 公开表": analysis["cai"]["public"],
                        "翻译一致": analysis["translation_matches_input"],
                        "酶切位点": len(analysis["restriction_sites"]),
                        "Motif 命中": len(analysis["motif_hits"]),
                        "局部 GC 警告": len(analysis["local_gc_outliers"]),
                        "非法碱基": ",".join(analysis["invalid_bases"]),
                    }
                )
        except Exception as exc:
            st.error(str(exc))
            return

        summary_rows.sort(key=lambda row: (row["基础问题"], row["风险警告"]), reverse=True)
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        st.download_button("下载 CDS 质检 CSV", data=records_to_csv(summary_rows), file_name="pichiaclm_cds_qc.csv")
        st.download_button(
            "下载 CDS 质检 JSON",
            data=json_dumps({"records": results}),
            file_name="pichiaclm_cds_qc.json",
            mime="application/json",
        )
        with st.expander("逐条 CDS 质检详情"):
            for result in results:
                render_cds_analysis_result(
                    result,
                    title=f"{result['id']} 质检详情",
                    key_prefix=f"external_batch_{result['id']}",
                )


def main() -> None:
    st.set_page_config(page_title="PichiaCLM", page_icon="DNA", layout="wide")
    st.title("PichiaCLM 氨基酸序列转 CDS")
    settings = sidebar_settings()
    tab_single, tab_batch, tab_external, tab_fusion = st.tabs(["单条预测", "FASTA 批量", "二次优化 CDS 质检", "信号肽拼接"])
    with tab_single:
        render_single_tab(settings)
    with tab_batch:
        render_batch_tab(settings)
    with tab_external:
        render_external_cds_tab(settings)
    with tab_fusion:
        render_fusion_tab(settings)


if __name__ == "__main__":
    main()
