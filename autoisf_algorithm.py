# aaps_emulator/core/autoisf_algorithm.py
"""
AutoISF algorithm utilities.

Этот модуль даёт «защитную» реализацию determine_basal_autoisf, которая:
 - принимает опциональный rt (runtime) и вытаскивает eventualBG, если он есть
 - безопасно обрабатывает отсутствие bg/delta
 - использует autosens ratio и sensitivityRatio из rt, если они есть
 - при необходимости конвертирует mg/dL -> mmol/L
 - считает insulinReq и переводит его в rate с защитами (duration, SMB scaling, лимиты)
 - приоритетно уважает insulinReq/rate из RT и сигналы RT об отключении SMB/базала
 - возвращает AutoIsfResult(eventualBG, insulinReq, rate, duration)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from aaps_emulator.core.cob_uam_pred import (
    build_pred_from_rt_lists,
    build_uam_pred,
    simple_cob_absorption,
)
from aaps_emulator.core.eventual_insulin_rate import (
    apply_basal_limits,
    combine_pred_curves,
    insulin_required_from_eventual,
    mgdl_to_mmol,
    rate_from_insulinReq,
)
from aaps_emulator.core.iob_openaps import InsulinEventSimple, compute_iob_openaps
from aaps_emulator.core.rt_parser import extract_lowtemp_rate, parse_rt_to_dict
from aaps_emulator.parsing.rt_parser import normalize_rt

logger = logging.getLogger(__name__)


@dataclass
class AutoIsfResult:
    eventualBG: float | None
    insulinReq: float | None
    rate: float
    duration: int
    smb: float | None = None


class TraceCollector:
    def __init__(self) -> None:
        self.steps: list[tuple[str, Any]] = []

    def add(self, name: str, value: Any) -> None:
        try:
            if hasattr(value, "__dict__"):
                value = value.__dict__
        except Exception as e:
            logger.debug("TraceCollector: suppressed exception: %s", e)
        self.steps.append((name, value))

    def dump(self) -> list[tuple[str, Any]]:
        return self.steps


def trace(tc: TraceCollector | None, name: str, value: Any) -> None:
    if tc is not None:
        tc.add(name, value)


def _extract_predicted_eventual_from_rt(rt_obj: Any) -> float | None:
    """
    Пытаемся вытащить eventualBG из rt (dict или строка).
    Возвращаем eventualBG в mmol/L или None.
    """
    if not rt_obj:
        return None

    if isinstance(rt_obj, dict):
        ev = rt_obj.get("eventualBG") or rt_obj.get("eventual_bg") or rt_obj.get("eventual")
        if ev is not None:
            try:
                evf = float(ev)
                return evf / 18.0 if evf > 30 else evf
            except Exception:
                logger.exception("autoisf_algorithm: suppressed exception")

        preds = rt_obj.get("predBGs") or rt_obj.get("predictions") or rt_obj.get("preds")
        if isinstance(preds, (list, tuple)) and preds:
            try:
                last = float(preds[-1])
                return last / 18.0 if last > 30 else last
            except Exception:
                logger.exception("autoisf_algorithm: suppressed exception in preds parsing")

        return None

    try:
        s = str(rt_obj)
        if "eventualBG=" in s:
            part = s.split("eventualBG=", 1)[1]
            num = ""
            for ch in part:
                if ch.isdigit() or ch in ".-,":
                    num += ch
                else:
                    break
            if num:
                val = float(num.replace(",", "."))
                return val / 18.0 if val > 30 else val

        m = re.search(r"eventual\s*bg\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)", s, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", "."))
            return val / 18.0 if val > 30 else val
    except Exception:
        logger.exception("autoisf_algorithm: suppressed exception in eventualBG projection")

    return None


def determine_basal_autoisf(
    glucose_status: Any,
    currenttemp: Any,
    iob_data_array: list[Any] | None,
    profile: Any,
    autosens_data: Any,
    meal_data: Any,
    rt: Any = None,
    microBolusAllowed: bool = False,
    currentTime: int = 0,
    flatBGsDetected: bool = False,
    autoIsfMode: bool = True,
    loop_wanted_smb: str = "none",
    profile_percentage: int = 100,
    smb_ratio: float = 0.5,
    smb_max_range_extension: float = 1.0,
    iob_threshold_percent: int = 100,
    auto_isf_consoleError: list[str] | None = None,
    auto_isf_consoleLog: list[str] | None = None,
    trace_mode: bool = False,
) -> AutoIsfResult | tuple[AutoIsfResult, list[tuple[str, Any]]]:
    """
    Defensive AutoISF calculation.
    """
    tc: TraceCollector | None = TraceCollector() if trace_mode else None

    # нормализуем rt в dict со snake_case и mmol
    try:
        if rt is not None:
            rt = normalize_rt(rt)
    except Exception as exc:
        logger.debug("normalize_rt not applied: %s", exc)

    if auto_isf_consoleError is None:
        auto_isf_consoleError = []
    if auto_isf_consoleLog is None:
        auto_isf_consoleLog = []

    res = AutoIsfResult(eventualBG=None, insulinReq=None, rate=0.0, duration=30)

    trace(tc, "input.glucose_status", glucose_status)
    trace(tc, "input.currenttemp", currenttemp)
    trace(tc, "input.iob_data_array", iob_data_array)
    trace(tc, "input.profile", profile)
    trace(tc, "input.autosens_data", autosens_data)
    trace(tc, "input.meal_data", meal_data)
    trace(tc, "input.rt", rt)

    if glucose_status is None or profile is None:
        auto_isf_consoleError.append("Missing glucose_status or profile")
        if trace_mode and tc is not None:
            return res, tc.dump()
        return res

    try:
        # bg и delta
        bg_val = getattr(glucose_status, "glucose", None)
        try:
            bg: float | None = float(bg_val) if bg_val is not None else None
        except (TypeError, ValueError):
            bg = None

        delta_raw = getattr(glucose_status, "delta", 0.0)
        try:
            delta: float = float(delta_raw or 0.0)
        except (TypeError, ValueError):
            delta = 0.0

        trace(tc, "bg", bg)
        trace(tc, "delta", delta)

        # === DELTA CAP (AAPS 3.4) ===
        # AAPS ограничивает влияние delta, чтобы не обнулять insulinReq при лёгком падении BG.
        try:
            d = float(delta)
        except Exception:
            d = 0.0

        # AAPS: если delta > -0.5 → считаем её нулевой
        capped_delta = 0.0 if d > -0.5 else d

        auto_isf_consoleLog.append(f"Delta-cap: raw_delta={d:.3f} → capped_delta={capped_delta:.3f}")

        delta = capped_delta
        trace(tc, "delta_capped", delta)

        # === LOW BG SAFETY (AAPS 3.4) ===
        # Если текущий BG ниже min_bg — AAPS запрещает дополнительный инсулин.
        min_bg_val = getattr(profile, "min_bg", None)
        try:
            min_bg = float(min_bg_val) if min_bg_val is not None else None
        except (TypeError, ValueError):
            min_bg = None

        # 1) eventualBG из rt
        eventualBG: float | None = None
        rt_obj: Any = rt

        # === TEMP BASAL INTERACTION (AAPS 3.4) ===
        # Если есть активный temp basal → AAPS НЕ включает safety и НЕ обнуляет insulinReq.
        temp_rate = None
        try:
            temp_rate = float(getattr(currenttemp, "rate", 0.0))
        except Exception:
            temp_rate = None

        temp_active = temp_rate is not None and temp_rate > 0

        if temp_active:
            auto_isf_consoleLog.append(f"Temp basal active ({temp_rate:.3f} U/h) → disabling low-BG and COB safety")
            disable_safety = True
        else:
            disable_safety = False

        trace(tc, "temp_active", temp_active)

        if not disable_safety and bg is not None and min_bg is not None and bg < min_bg:
            eventual_bg_low = _extract_predicted_eventual_from_rt(rt_obj)
            if eventual_bg_low is None:
                eventual_bg_low = bg
            auto_isf_consoleLog.append(
                f"Low BG safety: bg={bg:.3f} < min_bg={min_bg:.3f} → forcing insulinReq=0, rate=0"
            )
            res_low = AutoIsfResult(
                eventualBG=eventual_bg_low,
                insulinReq=0.0,
                rate=0.0,
                duration=30,
            )
            trace(tc, "low_bg_safety_result", res_low)
            if trace_mode and tc is not None:
                return res_low, tc.dump()
            return res_low

        # === MEAL / COB SAFETY (AAPS 3.4) ===
        # Если есть активные углеводы и BG падает — AAPS запрещает дозирование.
        meal_cob = getattr(meal_data, "meal_cob", 0) or 0
        cob = None
        if isinstance(rt_obj, dict):
            cob = rt_obj.get("cob") or rt_obj.get("meal_cob")

        if not disable_safety and (meal_cob > 0 or (cob is not None and cob > 0)) and delta is not None and delta < 0:
            auto_isf_consoleLog.append(
                f"Meal/COB safety: COB={meal_cob or cob}, delta={delta:.3f} → forcing insulinReq=0, rate=0"
            )
            res_meal = AutoIsfResult(
                eventualBG=eventualBG,
                insulinReq=0.0,
                rate=0.0,
                duration=30,
            )
            trace(tc, "meal_cob_safety_result", res_meal)
            if trace_mode and tc is not None:
                return res_meal, tc.dump()
            return res_meal

        # === NEGATIVE IOB SAFETY (AAPS 3.4) ===
        # Если IOB < 0 — AAPS разрешает дозирование даже при низком BG.
        iob_val = None
        if isinstance(rt_obj, dict):
            iob_val = rt_obj.get("iob")

        if iob_val is not None:
            try:
                iob_val = float(iob_val)
            except (TypeError, ValueError):
                iob_val = None

        if iob_val is not None and iob_val < 0:
            auto_isf_consoleLog.append(f"Negative IOB safety: iob={iob_val:.3f} → allow insulinReq even if BG < target")
            # Ничего не возвращаем — просто пропускаем safety-гейты

        # 1) eventualBG из rt
        eventualBG = None

        if rt_obj:
            ev_from_rt = _extract_predicted_eventual_from_rt(rt_obj)
            if ev_from_rt is not None:
                eventualBG = ev_from_rt
                auto_isf_consoleLog.append(f"Using predicted eventualBG from rt: {eventualBG:.3f} mmol/L")

        # 2) fallback: bg + delta*30
        if eventualBG is None:
            if bg is not None and bg > 50:
                bg = bg / 18.0
                auto_isf_consoleLog.append(f"Converted BG from mg/dL to mmol/L: {bg:.3f}")
                trace(tc, "bg_converted_mmol", bg)
            try:
                eventualBG = bg + delta * 30.0 if bg is not None else None
            except Exception:
                eventualBG = None

            if eventualBG is None or eventualBG <= 0:
                auto_isf_consoleLog.append(
                    f"Projected eventualBG invalid ({eventualBG}), falling back to current BG {bg}"
                )
                eventualBG = bg

        # --- ensure autosens_ratio is defined early ---
        autosens_ratio = 1.0
        try:
            if autosens_data is not None:
                r = getattr(autosens_data, "ratio", None)
                if r is not None:
                    autosens_ratio = float(r)
        except Exception:
            autosens_ratio = 1.0

        trace(tc, "autosens_ratio_early", autosens_ratio)

        trace(tc, "eventualBG", eventualBG)

        # ============================================================
        # AUTOISF: PREDICTED BG MODELS (UAM / COB / IOB)
        # ЭТАП 1 — расчёт предсказаний, но НЕ используем их для insulinReq/rate
        # ============================================================

        try:
            bg_val2 = getattr(glucose_status, "glucose", None)
            bg2 = float(bg_val2) if bg_val2 is not None else None
        except Exception:
            bg2 = None

        delta2 = delta  # уже капнутый delta
        short_avg_delta2 = getattr(glucose_status, "short_avg_delta", delta2)
        long_avg_delta2 = getattr(glucose_status, "long_avg_delta", delta2)

        sens2 = float(getattr(profile, "sens", 4.8) or 4.8)
        carb_ratio2 = float(getattr(profile, "carb_ratio", 15.0) or 15.0)

        meal_cob2 = float(getattr(meal_data, "meal_cob", 0.0) or 0.0)
        carbs2 = float(getattr(meal_data, "carbs", 0.0) or 0.0)

        # BGI = -activity * sens * 5
        try:
            activity2 = float(getattr(iob_data_array[0], "activity", 0.0) or 0.0)
        except Exception:
            activity2 = 0.0

        bgi2 = -activity2 * sens2 * 5.0

        # CI = carb impact
        ci2 = round((min(delta2, short_avg_delta2) - bgi2), 1)

        # CSF = sens / carb_ratio
        csf2 = sens2 / carb_ratio2 if carb_ratio2 > 0 else 0.0

        # Remaining carb absorption time (упрощённо)
        remainingCATime2 = 3.0 / autosens_ratio if autosens_ratio > 0 else 3.0

        # Total carb impact
        totalCI2 = max(0.0, ci2 / 5 * 60 * remainingCATime2 / 2)
        totalCA2 = totalCI2 / csf2 if csf2 > 0 else 0.0

        remainingCarbs2 = max(0.0, meal_cob2 - totalCA2)

        # Peak carb impact
        remainingCIpeak2 = remainingCarbs2 * csf2 * 5 / 60 / (remainingCATime2 / 2) if remainingCATime2 > 0 else 0.0

        # ------------------------------------------------------------
        # PREDICTED BG ARRAYS
        # ------------------------------------------------------------
        IOBpredBGs = [bg2] if bg2 is not None else [0.0]
        COBpredBGs = [bg2] if bg2 is not None else [0.0]
        UAMpredBGs = [bg2] if bg2 is not None else [0.0]

        slopeFromMaxDeviation2 = float(getattr(meal_data, "slope_from_max_deviation", 0.0) or 0.0)
        slopeFromMinDeviation2 = float(getattr(meal_data, "slope_from_min_deviation", 0.0) or 0.0)
        slopeFromDeviations2 = min(slopeFromMaxDeviation2, -slopeFromMinDeviation2 / 3)

        # Build predictions for 4 hours (48 steps)
        for i in range(48):
            predBGI2 = -activity2 * sens2 * 5.0
            predDev2 = ci2 * (1 - min(1.0, len(IOBpredBGs) / 12))
            IOBpredBGs.append(IOBpredBGs[-1] + predBGI2 + predDev2)

            predCI2 = max(0.0, ci2 * (1 - len(COBpredBGs) / max(1, remainingCATime2 * 12)))
            COBpredBGs.append(COBpredBGs[-1] + predBGI2 + predCI2)

            predUCI2 = max(0.0, ci2 + i * slopeFromDeviations2)
            UAMpredBGs.append(UAMpredBGs[-1] + predBGI2 + predUCI2)

        # ------------------------------------------------------------
        # EVENTUAL BG FALLBACK (НЕ ВЛИЯЕТ НА RATE НА ЭТАПЕ 1)
        # ------------------------------------------------------------
        eventualBG_uam2 = max(UAMpredBGs)
        eventualBG_cob2 = max(COBpredBGs)
        eventualBG_iob2 = max(IOBpredBGs)

        predicted_eventualBG2 = max(eventualBG, eventualBG_uam2, eventualBG_cob2)

        trace(tc, "predicted_eventualBG", predicted_eventualBG2)
        trace(tc, "UAMpredBG_last", UAMpredBGs[-1])
        trace(tc, "COBpredBG_last", COBpredBGs[-1])
        trace(tc, "IOBpredBG_last", IOBpredBGs[-1])

        # --- RT pre-checks and duration calculation (keep before diagnostics) ---
        parsed_rt: dict[str, Any] = parse_rt_to_dict(rt_obj)
        trace(tc, "parsed_rt", parsed_rt)

        lowtemp = extract_lowtemp_rate(parsed_rt)
        trace(tc, "lowtemp_rate", lowtemp)
        if lowtemp is not None:
            parsed_rt["rate"] = lowtemp
            auto_isf_consoleLog.append(f"Parsed low-temp rate from RT: {lowtemp:.3f} U/h")

        # === NEGATIVE IOB SAFETY (AAPS 3.4) ===
        # Если IOB < 0 → AAPS разрешает положительный basal даже при insulinReq == 0
        neg_iob_safety = False
        try:
            iob_rt = parsed_rt.get("iob")
            if iob_rt is None and isinstance(rt_obj, dict):
                iob_rt = rt_obj.get("iob")
            if iob_rt is not None and float(iob_rt) < 0:
                neg_iob_safety = True
                auto_isf_consoleLog.append(
                    f"Negative IOB safety: IOB={iob_rt} < 0 → allow basal even when insulinReq==0"
                )
        except Exception:
            neg_iob_safety = False

        trace(tc, "neg_iob_safety", neg_iob_safety)

        rt_disable_basal = False
        raw_text = str(parsed_rt.get("_raw_text", ""))

        if any(
            x in raw_text
            for x in [
                "disable smb",
                "disabling smb",
                "zero temp",
                "low temp",
                "microbolus",
                "min guard",
                "projected below",
            ]
        ):
            rt_disable_basal = True
            auto_isf_consoleLog.append("RT indicates SMB disabled / zero-temp → forcing basal=0 later")

        trace(tc, "rt_disable_basal", rt_disable_basal)

        # duration (compute early so diagnostics can use it)
        duration: int = 30
        try:
            if rt_obj and isinstance(rt_obj, dict):
                rt_dur = rt_obj.get("duration") or rt_obj.get("dur")
                if rt_dur is not None:
                    rd = int(float(rt_dur))
                    if rd > 300:
                        rd = max(5, rd // 60)
                    duration = max(5, rd)
                    auto_isf_consoleLog.append(f"Using duration from rt: {duration} minutes")
                else:
                    preds_rt = rt_obj.get("predBGs") or rt_obj.get("predictions")
                    if preds_rt and isinstance(preds_rt, (list, tuple)):
                        duration = max(duration, len(preds_rt) * 5)
        except Exception:
            logger.exception("autoisf_algorithm: suppressed exception in sensitivityRatio")

        try:
            if currenttemp is not None:
                dur_val = getattr(currenttemp, "duration", 30) or 30
                dur_int = int(dur_val)
                if dur_int > 0:
                    duration = max(duration, dur_int)
        except Exception:
            logger.exception("autoisf_algorithm: suppressed exception in duration fallback")

        if loop_wanted_smb and loop_wanted_smb != "none":
            duration = max(duration, 60)

        trace(tc, "duration", duration)
        # --- end RT pre-checks and duration ---

        # ------------------ Диагностический блок predBG / IOB / eventual → insulinReq → rate ------------------
        # переменная для итогового предсказанного eventualBG (COB/UAM/IOB-комбинация)
        predicted_eventualBG2: float | None = None

        # determine now_ts_ms safely from rt timestamp if present
        now_ts_ms = int(time.time() * 1000)
        try:
            if isinstance(rt_obj, dict):
                ts_raw = rt_obj.get("timestamp")
                if ts_raw is not None:
                    try:
                        now_ts_ms = int(float(ts_raw))
                    except (TypeError, ValueError):
                        now_ts_ms = int(time.time() * 1000)
        except Exception:
            now_ts_ms = int(time.time() * 1000)

        # IOB OpenAPS (диагностика)
        try:
            simple_events: list[InsulinEventSimple] = []
            for e in iob_data_array or []:
                ts_e = getattr(e, "timestamp", None) or getattr(e, "date", None) or now_ts_ms
                amt = float(getattr(e, "amount", 0.0) or 0.0)
                dur_e = int(getattr(e, "duration", 0) or 0)
                rate_e = float(getattr(e, "rate", 0.0) or 0.0)
                typ = str(getattr(e, "type", "bolus") or "bolus")
                simple_events.append(InsulinEventSimple(ts_e, amt, dur_e, rate_e, typ))
            iob_info = compute_iob_openaps(simple_events, now_ts_ms, dia_hours=4.0)
            trace(tc, "diagnostic.iob_openaps", iob_info.get("iob"))
            trace(tc, "diagnostic.activity_openaps", iob_info.get("activity"))
        except Exception as _e:
            trace(tc, "diagnostic.iob_openaps_error", str(_e))

        # preds from RT or built from meal/profile
        preds: dict[str, Any] = {}
        try:
            preds = build_pred_from_rt_lists(rt_obj or {})
            trace(
                tc,
                "diagnostic.preds_from_rt_lengths",
                {k: len(v) for k, v in preds.items() if hasattr(v, "__len__")},
            )
        except Exception:
            preds = {"pred_iob": [], "pred_cob": [], "pred_uam": [], "pred_zt": []}

        # build COB if missing
        ci_per_5m: float = 0.0
        try:
            if not preds.get("pred_cob"):
                if meal_data and getattr(meal_data, "meal_cob", None):
                    cob_model = simple_cob_absorption(
                        meal_cob_g=float(getattr(meal_data, "meal_cob", 0.0) or 0.0),
                        cat_hours=2.0,
                        step_minutes=5,
                    )
                    cr = getattr(profile, "carb_ratio", None)
                    if cr and float(cr) > 0:
                        cr_f = float(cr)
                        sens_mmol = float(getattr(profile, "sens", 4.8) or 4.8)
                        sens_mgdl_per_U = sens_mmol * 18.0
                        mgdl_per_gram = sens_mgdl_per_U / cr_f
                        preds["pred_cob"] = [float(g) * mgdl_per_gram for g in cob_model["pred_cob"]]
                        ci_per_5m = float(cob_model["ci_per_5m"]) * mgdl_per_gram
                    else:
                        preds["pred_cob"] = []
                        ci_per_5m = 0.0
                else:
                    ci_per_5m = 0.0
            else:
                try:
                    pred_cob_list = preds.get("pred_cob", [])
                    ci_per_5m = float(pred_cob_list[0] or 0.0) if pred_cob_list else 0.0
                except Exception:
                    ci_per_5m = 0.0
            trace(tc, "diagnostic.ci_per_5m", ci_per_5m)
        except Exception as _e:
            trace(tc, "diagnostic.cob_build_error", str(_e))

        # UAM pred
        try:
            if not preds.get("pred_uam"):
                uam_impact: float | None = None
                uam_duration: float | None = None
                if isinstance(rt_obj, dict):
                    uam_impact_raw = rt_obj.get("uam_impact") or rt_obj.get("uamImpact") or rt_obj.get("uamimpact")
                    uam_duration_raw = rt_obj.get("uam_duration_hours") or rt_obj.get("uamDuration")
                    if uam_impact_raw is not None:
                        uam_impact = float(uam_impact_raw)
                    if uam_duration_raw is not None:
                        uam_duration = float(uam_duration_raw)
                if uam_impact is not None:
                    preds["pred_uam"] = build_uam_pred(
                        uam_impact, uam_duration if uam_duration is not None else 3.0, step_minutes=5
                    )
                else:
                    preds["pred_uam"] = []
            trace(tc, "diagnostic.pred_uam_len", len(preds.get("pred_uam", [])))
        except Exception as _e:
            trace(tc, "diagnostic.uam_build_error", str(_e))

        # combine and eventual (diagnostic)
        try:
            combined = combine_pred_curves(preds)
            eventual_mgdl_calc_any = combined.get("eventual_mgdl", 0.0)
            try:
                eventual_mgdl_calc = float(eventual_mgdl_calc_any)
            except (TypeError, ValueError):
                eventual_mgdl_calc = 0.0
            eventual_mmol_calc = mgdl_to_mmol(eventual_mgdl_calc)
            trace(tc, "diagnostic.eventual_mgdl_calc", eventual_mgdl_calc)
            trace(tc, "diagnostic.eventual_mmol_calc", eventual_mmol_calc)
        except Exception as _e:
            eventual_mgdl_calc = None
            eventual_mmol_calc = None
            trace(tc, "diagnostic.combine_pred_error", str(_e))

        # сохраняем комбинированный eventual как predicted_eventualBG2 для fallback
        try:
            if eventual_mmol_calc is not None:
                predicted_eventualBG2 = float(eventual_mmol_calc)
                trace(tc, "diagnostic.predicted_eventualBG2", predicted_eventualBG2)
        except Exception as _e:
            trace(tc, "diagnostic.predicted_eventualBG2_error", str(_e))

        # if eventualBG not provided by RT, use diagnostic eventual
        try:
            if eventualBG is None and eventual_mmol_calc is not None:
                eventualBG = eventual_mmol_calc
                trace(tc, "diagnostic.eventualBG_from_combined", eventualBG)
        except Exception as e:
            logger.debug("autoisf_algorithm: suppressed exception in diagnostic.eventualBG_from_combined: %s", e)
            trace(tc, "diagnostic.eventualBG_from_combined_error", str(e))

        # diagnostic insulinReq and rate
        try:
            effective_sens_diag_raw = getattr(profile, "variable_sens", getattr(profile, "sens", 4.8))
            effective_sens_diag = float(effective_sens_diag_raw or 4.8)
            if eventualBG is not None:
                insulinReq_calc = insulin_required_from_eventual(
                    eventualBG,
                    float(getattr(profile, "target_bg", 6.4) or 6.4),
                    effective_sens_diag,
                )
            else:
                insulinReq_calc = None
            trace(tc, "diagnostic.insulinReq_calc", insulinReq_calc)

            duration_for_rate = int(getattr(currenttemp, "duration", 30) or 30) if currenttemp else 30
            rate_calc = (
                rate_from_insulinReq(float(insulinReq_calc or 0.0), duration_for_rate)
                if insulinReq_calc is not None
                else 0.0
            )

            # diagnostic limits: raw (no max_daily) and strict (respect max_daily)
            limits_calc = apply_basal_limits(rate_calc, profile, respect_max_daily=False)
            trace(tc, "diagnostic.rate_calc_raw", rate_calc)
            trace(tc, "diagnostic.limits_calc", limits_calc)

            limits_calc_strict = apply_basal_limits(rate_calc, profile, respect_max_daily=True)
            trace(tc, "diagnostic.limits_calc_strict", limits_calc_strict)
        except Exception as _e:
            trace(tc, "diagnostic.insulin_rate_calc_error", str(_e))
        # ------------------ конец диагностического блока ------------------

        # ============================================================
        # AUTOISF: eventualBG fallback (ЭТАП 2)
        # Используем предсказания COB/UAM, если они выше RT eventualBG
        # ============================================================

        try:
            if predicted_eventualBG2 is not None and eventualBG is not None:
                eventualBG = max(eventualBG, predicted_eventualBG2)
                auto_isf_consoleLog.append(
                    f"eventualBG fallback: using max(RT={eventualBG:.3f}, predicted={predicted_eventualBG2:.3f})"
                )
                trace(tc, "eventualBG_after_fallback", eventualBG)
        except Exception as e:
            auto_isf_consoleLog.append(f"eventualBG fallback error: {e}")
            trace(tc, "eventualBG_fallback_error", str(e))

        # target
        target_bg_raw = getattr(profile, "target_bg", None) or 6.4
        target_bg = float(target_bg_raw)
        trace(tc, "target_bg", target_bg)

        # ============================================================
        # AUTOISF: minPredBG / minGuardBG (ЭТАП 3)
        # Полная safety‑логика AAPS для минимального прогнозируемого BG
        # ============================================================

        try:
            # Минимальные значения по кривым
            minIOBPredBG = min(IOBpredBGs) if IOBpredBGs else float("inf")
            minCOBPredBG = min(COBpredBGs) if COBpredBGs else float("inf")
            minUAMPredBG = min(UAMpredBGs) if UAMpredBGs else float("inf")

            # Zero‑temp guard BG (пока используем IOBpredBGs как приближение)
            minZTGuardBG = minIOBPredBG

            # minGuardBG = минимальный BG среди всех защитных кривых
            minGuardBG = min(minIOBPredBG, minCOBPredBG, minUAMPredBG, minZTGuardBG)

            # fractionCarbsLeft = COB / carbs
            try:
                fractionCarbsLeft = meal_cob2 / carbs2 if carbs2 > 0 else 0.0
            except Exception:
                fractionCarbsLeft = 0.0

            # minZTUAMPredBG — сглаживание UAM через ZT
            minZTUAMPredBG = minUAMPredBG
            if minZTGuardBG < min_bg:
                minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0
            elif minZTGuardBG < target_bg:
                blendPct = (minZTGuardBG - min_bg) / (target_bg - min_bg) if (target_bg - min_bg) != 0 else 1.0
                blended = minUAMPredBG * blendPct + minZTGuardBG * (1 - blendPct)
                minZTUAMPredBG = (minUAMPredBG + blended) / 2.0
            elif minZTGuardBG > minUAMPredBG:
                minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0

            # blendedMinPredBG — смешивание COB/UAM
            if minCOBPredBG < float("inf") and minUAMPredBG < float("inf"):
                blendedMinPredBG = fractionCarbsLeft * minCOBPredBG + (1 - fractionCarbsLeft) * minZTUAMPredBG
            else:
                blendedMinPredBG = minZTUAMPredBG

            # minPredBG — итоговый минимальный прогноз BG
            minPredBG = min(minIOBPredBG, blendedMinPredBG, minGuardBG)

            trace(tc, "minIOBPredBG", minIOBPredBG)
            trace(tc, "minCOBPredBG", minCOBPredBG)
            trace(tc, "minUAMPredBG", minUAMPredBG)
            trace(tc, "minZTGuardBG", minZTGuardBG)
            trace(tc, "minGuardBG", minGuardBG)
            trace(tc, "minZTUAMPredBG", minZTUAMPredBG)
            trace(tc, "blendedMinPredBG", blendedMinPredBG)
            trace(tc, "minPredBG", minPredBG)

        except Exception as e:
            auto_isf_consoleLog.append(f"minPredBG error: {e}")
            trace(tc, "minPredBG_error", str(e))
            minPredBG = None
            minGuardBG = None

        # autosens ratio
        autosens_ratio: float = 1.0
        if autosens_data is not None:
            try:
                r = getattr(autosens_data, "ratio", None)
                if r is not None and r != 1.0:
                    autosens_ratio = float(r)
            except Exception:
                autosens_ratio = 1.0

        # sensitivityRatio из rt
        try:
            sens_ratio_rt: float | None = None
            if rt_obj and isinstance(rt_obj, dict):
                sens_ratio_rt_raw = (
                    rt_obj.get("sensitivityRatio")
                    or rt_obj.get("sensitivity_ratio")
                    or rt_obj.get("sensitivityRatioValue")
                )
                if sens_ratio_rt_raw is not None:
                    sens_ratio_rt = float(sens_ratio_rt_raw)
            if sens_ratio_rt is not None:
                autosens_ratio = float(sens_ratio_rt)
                auto_isf_consoleLog.append(f"Using sensitivityRatio from rt: {autosens_ratio:.3f}")
        except Exception:
            logger.exception("autoisf_algorithm: suppressed exception in duration calculation")

        trace(tc, "autosens_ratio", autosens_ratio)

        # чувствительность
        prof_sens_raw = getattr(profile, "variable_sens", None) or getattr(profile, "sens", None)
        if prof_sens_raw is None or prof_sens_raw == 0:
            prof_sens_raw = 6.0
        prof_sens = round(float(prof_sens_raw), 3)
        effective_sens = round(prof_sens * autosens_ratio, 4)
        if effective_sens == 0:
            effective_sens = prof_sens or 6.0

        trace(tc, "prof_sens", prof_sens)
        trace(tc, "effective_sens", effective_sens)

        # ============================================================
        # AUTOISF: insulinReq через min(minPredBG, eventualBG) (ЭТАП 4)
        # ============================================================

        insulinReq = None
        try:
            if eventualBG is not None:
                # если minPredBG посчитан — используем min(minPredBG, eventualBG)
                if "minPredBG" in locals() and minPredBG is not None:
                    bg_for_ins_req = min(minPredBG, eventualBG)
                else:
                    bg_for_ins_req = eventualBG

                insulinReq = insulin_required_from_eventual(
                    bg_for_ins_req,
                    target_bg,
                    effective_sens,
                )

                trace(tc, "insulinReq_bg_for_calc", bg_for_ins_req)
                trace(tc, "insulinReq_effective_sens", effective_sens)
            else:
                insulinReq = None
        except Exception as e:
            trace(tc, "insulinReq_calc_error_main", str(e))
            insulinReq = None

        trace(tc, "insulinReq_raw", insulinReq)

        # ============================================================
        # AUTOISF: LGS / Zero-temp при угрозе гипо (ЭТАП 5.1)
        # ============================================================

        try:
            if minGuardBG is not None and min_bg is not None and minGuardBG < min_bg:
                auto_isf_consoleLog.append(
                    f"LGS: minGuardBG={minGuardBG:.3f} < min_bg={min_bg:.3f} → forcing zero-temp"
                )
                insulinReq = 0.0
                rate = 0.0
                duration = 30
                res_lgs = AutoIsfResult(
                    eventualBG=eventualBG,
                    insulinReq=0.0,
                    rate=0.0,
                    duration=30,
                )
                trace(tc, "lgs_result", res_lgs)
                if trace_mode and tc is not None:
                    return res_lgs, tc.dump()
                return res_lgs
        except Exception as e:
            trace(tc, "lgs_error", str(e))

        # ============================================================
        # AUTOISF: minutesAboveMinBG / minutesAboveThreshold (ЭТАП 5.2)
        # ============================================================

        try:
            minutesAboveMinBG = 0
            minutesAboveThreshold = 0

            # каждая точка предсказания — 5 минут
            for bg_val in IOBpredBGs:
                if bg_val > min_bg:
                    minutesAboveMinBG += 5
                if bg_val > target_bg:
                    minutesAboveThreshold += 5

            trace(tc, "minutesAboveMinBG", minutesAboveMinBG)
            trace(tc, "minutesAboveThreshold", minutesAboveThreshold)

        except Exception as e:
            trace(tc, "minutesAboveMinBG_error", str(e))
            minutesAboveMinBG = 0
            minutesAboveThreshold = 0

        # ============================================================
        # AUTOISF: carbsReq (ЭТАП 5.3)
        # ============================================================

        try:
            carbsReq = 0.0
            if minPredBG is not None and min_bg is not None and minPredBG < min_bg:
                # CSF = sens / carb_ratio
                try:
                    csf = float(getattr(profile, "sens", 4.8)) / float(getattr(profile, "carb_ratio", 15.0))
                except Exception:
                    csf = 0.3  # fallback

                carbsReq = max(0.0, (min_bg - minPredBG) / csf)
                auto_isf_consoleLog.append(
                    f"carbsReq: minPredBG={minPredBG:.3f} < min_bg={min_bg:.3f} → carbsReq={carbsReq:.1f}g"
                )

            trace(tc, "carbsReq", carbsReq)

        except Exception as e:
            trace(tc, "carbsReq_error", str(e))
            carbsReq = 0.0

        # ============================================================
        # AUTOISF: SMB eligibility (ЭТАП 6.1)
        # ============================================================

        try:
            allowSMB = False

            # 1) SMB must be enabled in profile
            smb_always = bool(getattr(profile, "enableSMB_always", False))
            smb_with_cob = bool(getattr(profile, "enableSMB_with_COB", False))
            smb_after_carbs = bool(getattr(profile, "enableSMB_after_carbs", False))
            smb_with_temptarget = bool(getattr(profile, "enableSMB_with_temptarget", False))
            allow_high_tt = bool(getattr(profile, "allowSMB_with_high_temptarget", False))

            # 2) COB condition
            has_cob = meal_cob2 > 0

            # 3) carbs recently eaten
            carbs_recent = (now_ts_ms - last_carb_time2) < 3 * 60 * 60 * 1000  # 3 часа

            # 4) temptarget condition
            temptarget_active = (target_bg != min_bg) or (target_bg != max_bg)
            temptarget_high = target_bg > max_bg

            # 5) BG must be above min_bg
            bg_ok = bg > min_bg

            # 6) LGS must NOT be active
            lgs_active = minGuardBG is not None and minGuardBG < min_bg

            # 7) negIOB safety
            neg_iob_active = neg_iob_safety

            # 8) RT override disables SMB
            rt_override_active = rt_rate is not None

            # === AAPS SMB eligibility logic ===
            if not lgs_active and not neg_iob_active and not rt_override_active and bg_ok:
                if smb_always or smb_with_cob and has_cob or smb_after_carbs and carbs_recent:
                    allowSMB = True
                elif smb_with_temptarget and temptarget_active:
                    allowSMB = not (temptarget_high and not allow_high_tt)

            trace(tc, "allowSMB", allowSMB)
            trace(tc, "smb_always", smb_always)
            trace(tc, "smb_with_cob", smb_with_cob)
            trace(tc, "smb_after_carbs", smb_after_carbs)
            trace(tc, "smb_with_temptarget", smb_with_temptarget)
            trace(tc, "temptarget_active", temptarget_active)
            trace(tc, "temptarget_high", temptarget_high)
            trace(tc, "bg_ok", bg_ok)
            trace(tc, "lgs_active", lgs_active)
            trace(tc, "neg_iob_active", neg_iob_active)
            trace(tc, "rt_override_active", rt_override_active)

        except Exception as e:
            trace(tc, "allowSMB_error", str(e))
            allowSMB = False

        # ============================================================
        # AUTOISF: SMB limits (ЭТАП 6.2)
        # ============================================================

        try:
            # базовые лимиты из профиля
            maxSMBBasalMinutes = float(getattr(profile, "maxSMBBasalMinutes", 30.0))
            maxUAMSMBBasalMinutes = float(getattr(profile, "maxUAMSMBBasalMinutes", 30.0))

            # коэффициенты SMB
            smb_ratio_min = float(getattr(profile, "smb_delivery_ratio_min", 0.5))
            smb_ratio_max = float(getattr(profile, "smb_delivery_ratio_max", 0.6))
            smb_bg_range = float(getattr(profile, "smb_delivery_ratio_bg_range", 0.0))
            smb_range_extension = float(getattr(profile, "smb_max_range_extension", 1.0))

            # IOB thresholds
            try:
                iobThreshold = float(getattr(profile, "iob_threshold", 0.0))
            except Exception:
                iobThreshold = 0.0

            try:
                iobThresholdVirtual = float(getattr(profile, "iob_threshold_virtual", 0.0))
            except Exception:
                iobThresholdVirtual = 0.0

            # SMB ограничение по IOB
            iob_ok = True
            if iob is not None:
                if iobThreshold > 0 and iob > iobThreshold:
                    iob_ok = False
                if iobThresholdVirtual > 0 and iob > iobThresholdVirtual:
                    iob_ok = False

            # SMB ограничение по BG (расширение диапазона)
            smb_ratio_dynamic = smb_ratio_min
            if smb_bg_range > 0:
                bg_diff = max(0.0, bg - target_bg)
                extension = min(1.0, bg_diff / smb_bg_range)
                smb_ratio_dynamic = smb_ratio_min + extension * (smb_ratio_max - smb_ratio_min)
            else:
                smb_ratio_dynamic = smb_ratio_min

            # итоговый коэффициент SMB
            smb_ratio_dynamic = max(smb_ratio_min, min(smb_ratio_dynamic, smb_ratio_max))

            trace(tc, "maxSMBBasalMinutes", maxSMBBasalMinutes)
            trace(tc, "maxUAMSMBBasalMinutes", maxUAMSMBBasalMinutes)
            trace(tc, "smb_ratio_min", smb_ratio_min)
            trace(tc, "smb_ratio_max", smb_ratio_max)
            trace(tc, "smb_ratio_dynamic", smb_ratio_dynamic)
            trace(tc, "iobThreshold", iobThreshold)
            trace(tc, "iobThresholdVirtual", iobThresholdVirtual)
            trace(tc, "iob_ok", iob_ok)

        except Exception as e:
            trace(tc, "smb_limits_error", str(e))
            maxSMBBasalMinutes = 0.0
            maxUAMSMBBasalMinutes = 0.0
            smb_ratio_dynamic = 0.0
            iob_ok = False

        # ============================================================
        # AUTOISF: SMB calculation (ЭТАП 6.3)
        # ============================================================

        smb = None
        try:
            # SMB возможен только если allowSMB == True
            if allowSMB and insulinReq is not None and insulinReq > 0 and iob_ok:
                # переводим insulinReq в эквивалент базала
                # AAPS: smbBasalReq = insulinReq * (60 / duration)
                try:
                    dur = float(duration)
                except Exception:
                    dur = 30.0

                smbBasalReq = float(insulinReq) * (60.0 / max(1.0, dur))
                trace(tc, "smbBasalReq", smbBasalReq)

                # ограничение maxSMBBasalMinutes
                max_smb_units = (maxSMBBasalMinutes / 60.0) * current_basal
                trace(tc, "max_smb_units", max_smb_units)

                # ограничение maxUAMSMBBasalMinutes (если UAM активно)
                if enable_uam_flag:
                    max_smb_units = min(max_smb_units, (maxUAMSMBBasalMinutes / 60.0) * current_basal)
                    trace(tc, "max_smb_units_after_uam", max_smb_units)

                # ограничение smb_ratio_dynamic
                smb_units = smbBasalReq * smb_ratio_dynamic
                trace(tc, "smb_units_before_clamp", smb_units)

                # финальный clamp
                smb_units = max(0.0, min(smb_units, max_smb_units))
                trace(tc, "smb_units_after_clamp", smb_units)

                # минимальный порог SMB (AAPS: 0.05 U)
                if smb_units < 0.05:
                    smb_units = 0.0

                if smb_units > 0:
                    smb = round(smb_units, 3)
                    auto_isf_consoleLog.append(f"SMB: smb={smb:.3f}U (ratio={smb_ratio_dynamic:.3f})")
                else:
                    smb = None

            else:
                trace(tc, "smb_skipped", f"allowSMB={allowSMB}, insulinReq={insulinReq}, iob_ok={iob_ok}")

        except Exception as e:
            trace(tc, "smb_calc_error", str(e))
            smb = None

        # rate
        rate: float = 0.0
        try:
            # === FIX 1: корректная проверка наличия ключей rate/deliveryRate ===
            rt_rate_provided_flag = None

            # 1) parsed_rt
            if isinstance(parsed_rt, dict):
                if "rate" in parsed_rt:
                    rt_rate_provided_flag = parsed_rt["rate"]
                elif "deliveryRate" in parsed_rt:
                    rt_rate_provided_flag = parsed_rt["deliveryRate"]

            # 2) сырой rt
            if rt_rate_provided_flag is None and isinstance(rt_obj, dict):
                if "rate" in rt_obj:
                    rt_rate_provided_flag = rt_obj["rate"]
                elif "deliveryRate" in rt_obj:
                    rt_rate_provided_flag = rt_obj["deliveryRate"]

            try:
                rt_rate_val = float(rt_rate_provided_flag) if rt_rate_provided_flag is not None else None
            except Exception:
                rt_rate_val = None

            trace(tc, "rt_rate_val", rt_rate_val)

            trace(tc, "rt_rate_val", rt_rate_val)

            if insulinReq is None:
                raw_rate = 0.0
            else:
                if float(insulinReq) <= 0.0 and rt_rate_val is None:
                    raw_rate = 0.0
                else:
                    if rt_rate_val is not None:
                        raw_rate = float(rt_rate_val)
                        auto_isf_consoleLog.append(
                            f"Using RT explicit rate as raw_rate: {raw_rate:.3f} U/h (skip delta cap and SMB scaling)"
                        )
                    else:
                        raw_rate = float(insulinReq) * (60.0 / max(1, duration))

            max_basal = getattr(profile, "max_basal", None)
            if max_basal is not None:
                try:
                    raw_rate = min(raw_rate, float(max_basal))
                except Exception as e:
                    logger.debug("autoisf_algorithm: failed to apply max_basal=%r: %s", max_basal, e)

            trace(tc, "raw_rate_after_max_basal", raw_rate)

            if rt_rate_val is None:
                current_basal = float(getattr(profile, "current_basal", 0.0) or 0.0)
                try:
                    profile_max_delta = getattr(profile, "max_delta_rate", None)
                    max_delta_rate = float(profile_max_delta) if profile_max_delta is not None else 2.0
                except Exception:
                    max_delta_rate = 2.0
                allowed_max = current_basal + max_delta_rate
                raw_rate = min(raw_rate, allowed_max)

                trace(tc, "allowed_max", allowed_max)
                trace(tc, "raw_rate_after_allowed_max", raw_rate)

            smb_threshold = 1.0
            if (
                insulinReq is not None
                and abs(insulinReq) < smb_threshold
                and bool(getattr(profile, "enableSMB_always", False))
                and rt_rate_val is None
            ):
                smb_ratio_local = getattr(profile, "smb_delivery_ratio", smb_ratio or 0.5)
                raw_rate = raw_rate * float(smb_ratio_local)
                trace(tc, "raw_rate_after_smb_scaling", raw_rate)
            elif rt_rate_val is not None:
                auto_isf_consoleLog.append(
                    "RT rate present — delta cap and SMB scaling skipped to preserve RT decision"
                )

            try:
                # корректная проверка наличия флага rate в parsed_rt (даже если 0.0)
                if isinstance(parsed_rt, dict):
                    if "rate" in parsed_rt or "deliveryRate" in parsed_rt:
                        rt_rate_provided_flag_check = (
                            parsed_rt.get("rate") if "rate" in parsed_rt else parsed_rt.get("deliveryRate")
                        )
                    else:
                        rt_rate_provided_flag_check = None
                else:
                    rt_rate_provided_flag_check = None
            except Exception:
                rt_rate_provided_flag_check = None

            trace(tc, "rt_rate_provided_flag_check", rt_rate_provided_flag_check)

            if (
                insulinReq is not None
                and insulinReq == 0.0
                and not rt_disable_basal
                and not rt_rate_provided_flag_check
                and not neg_iob_safety
            ):
                raw_rate = 0.0
                auto_isf_consoleLog.append(
                    "insulinReq==0.0 and no RT rate provided — forcing raw_rate=0.0 (no negative IOB safety)"
                )
                trace(tc, "raw_rate_forced_zero_by_insulinReq", raw_rate)

            # === FIX 2: final_rate_source должен учитывать наличие ключа, а не truthiness ===
            final_rate_source = "computed"

            if isinstance(parsed_rt, dict) and ("rate" in parsed_rt or "deliveryRate" in parsed_rt):
                final_rate_source = "rt_rate"

            if isinstance(rt_obj, dict) and ("rate" in rt_obj or "deliveryRate" in rt_obj):
                final_rate_source = "rt_rate"

            trace(tc, "final_rate_source", final_rate_source)

            trace(tc, "final_rate_source", final_rate_source)

            auto_isf_consoleLog.append(
                f"Final rate decision (pre-clamp): raw_rate={raw_rate:.3f} source={final_rate_source}"
            )

            # === AAPS RT BASAL OVERRIDE (correct placement) ===
            # === FIX 3: корректная проверка наличия ключей rate/duration ===
            rt_rate = None
            rt_dur = None

            # 1) parsed_rt
            if isinstance(parsed_rt, dict):
                if "rate" in parsed_rt:
                    rt_rate = parsed_rt["rate"]
                elif "deliveryRate" in parsed_rt:
                    rt_rate = parsed_rt["deliveryRate"]
                rt_dur = parsed_rt.get("duration") or parsed_rt.get("dur")

            # 2) сырой rt
            if rt_rate is None and isinstance(rt_obj, dict):
                if "rate" in rt_obj:
                    rt_rate = rt_obj["rate"]
                elif "deliveryRate" in rt_obj:
                    rt_rate = rt_obj["deliveryRate"]
                if rt_dur is None:
                    rt_dur = rt_obj.get("duration") or rt_obj.get("dur")

            # convert
            try:
                rt_rate = float(rt_rate) if rt_rate is not None else None
            except Exception:
                rt_rate = None

            try:
                rt_dur = int(float(rt_dur)) if rt_dur is not None else None
            except Exception:
                rt_dur = None

            trace(tc, "rt_rate", rt_rate)
            trace(tc, "rt_dur", rt_dur)

            # AAPS override conditions
            rt_override_allowed = (
                rt_rate is not None
                and rt_dur is not None
                and rt_dur > 0
                and not rt_disable_basal
                and not neg_iob_safety  # AAPS still allows override even with neg IOB, but only if rate>0
            )

            if rt_override_allowed and rt_rate is not None:
                auto_isf_consoleLog.append(f"RT override: using temp basal {rt_rate:.3f} U/h for {rt_dur} min")

            try:
                # округлим insulinReq для логов, если он мал
                if insulinReq is not None and abs(insulinReq) < smb_threshold:
                    insulinReq = round(float(insulinReq), 3)
                trace(tc, "insulinReq_rounded", insulinReq)

                # округлим raw_rate для логов
                raw_rate = round(float(raw_rate), 3)
                trace(tc, "raw_rate_rounded", raw_rate)

                # === MAX DELTA RATE LIMIT (AAPS 3.4) ===
                # Ограничиваем скорость изменения базала, чтобы не прыгать слишком резко.
                try:
                    current_basal_md = float(profile.current_basal)
                except Exception:
                    current_basal_md = 0.0

                # Максимальное изменение за 5 минут (как в AAPS)
                max_delta_per_5min = 0.3  # AAPS internal default

                # duration в минутах
                try:
                    dur = float(duration)
                except Exception:
                    dur = 30.0

                # сколько 5-минутных интервалов
                intervals = max(1.0, dur / 5.0)

                # максимальное изменение за duration
                max_delta_rate_md = max_delta_per_5min * intervals

                # ограничиваем raw_rate
                delta_md = raw_rate - current_basal_md
                if delta_md > max_delta_rate_md:
                    raw_rate = current_basal_md + max_delta_rate_md
                    auto_isf_consoleLog.append(
                        f"max_delta_rate limit: reducing raw_rate to {raw_rate:.3f} (max_delta={max_delta_rate_md:.3f})"
                    )
                elif delta_md < -max_delta_rate_md:
                    raw_rate = current_basal_md - max_delta_rate_md
                    auto_isf_consoleLog.append(
                        f"max_delta_rate limit: increasing raw_rate to {raw_rate:.3f} "
                        f"(max_delta={max_delta_rate_md:.3f})"
                    )

                trace(tc, "raw_rate_after_max_delta", raw_rate)

                # применяем строгие лимиты для финального решения (учитываем max_daily_basal)
                final_limits = apply_basal_limits(raw_rate, profile, respect_max_daily=True)
                trace(tc, "final_limits_applied", final_limits)

                final_rate_val = final_limits.get("final_rate", 0.0)
                final_rate = float(final_rate_val)
                rate = round(max(0.0, final_rate), 2)
                trace(tc, "final_rate", rate)

                # === STRICT MAX DAILY BASAL LIMIT (AAPS 3.4) ===
                try:
                    max_daily = float(getattr(profile, "max_daily_basal", None) or 0.0)
                except Exception:
                    max_daily = 0.0

                if max_daily > 0.0 and rate > max_daily:
                    auto_isf_consoleLog.append(f"max_daily_basal limit: rate {rate:.2f} → {max_daily:.2f}")
                    rate = max_daily
                    trace(tc, "rate_after_max_daily_basal", rate)

                # === SMB-ONLY MODE (AAPS 3.4) ===
                # Если AAPS в SMB-only режиме — basal=0, insulinReq сохраняется.
                smb_only = False
                if isinstance(rt_obj, dict):
                    smb_only = bool(rt_obj.get("smb_only") or rt_obj.get("smbOnly"))

                if smb_only:
                    auto_isf_consoleLog.append("SMB-only mode → forcing basal=0")
                    rate = 0.0
                    trace(tc, "smb_only_rate_zero", rate)

                auto_isf_consoleLog.append(f"Final rate after strict clamps: rate={rate:.2f} U/h (raw={raw_rate:.3f})")
            except Exception:
                # fallback: если что-то пошло не так, используем безопасный raw_rate
                rate = max(0.0, raw_rate)
                trace(tc, "final_rate_exception_fallback", rate)

        except Exception:
            rate = 0.0
            trace(tc, "final_rate_exception_total", rate)

        try:
            if rt_disable_basal:
                auto_isf_consoleLog.append(
                    "Enforcing RT disable: overriding final rate -> 0.0 U/h (insulinReq preserved)"
                )
                rate = 0.0
                trace(tc, "final_rate_rt_disable_basal", rate)
        except Exception:
            logger.exception("autoisf_algorithm: suppressed exception in RT disable basal")

        res.eventualBG = eventualBG
        res.insulinReq = insulinReq
        res.rate = rate
        res.duration = duration
        res.smb = smb

        trace(tc, "result.eventualBG", res.eventualBG)
        trace(tc, "result.insulinReq", res.insulinReq)
        trace(tc, "result.rate", res.rate)
        trace(tc, "result.duration", res.duration)

        # mark final pred lists (if present) with final. prefix to avoid confusion with diagnostic.*
        try:
            trace(tc, "final.pred_uam_first10", preds.get("pred_uam", [])[:10])
            trace(tc, "final.pred_zt_first10", preds.get("pred_zt", [])[:10])
        except Exception as e:
            logger.debug("autoisf_algorithm: suppressed exception while tracing final preds: %s", e)
            trace(tc, "final.pred_trace_error", str(e))

        bg_str = f"{bg:.2f}" if (bg is not None) else "None"
        ev_str = f"{eventualBG:.2f}" if (eventualBG is not None) else "None"
        ins_str = f"{insulinReq:.3f}" if (insulinReq is not None) else "None"

        auto_isf_consoleLog.append(
            f"AutoISF calc: bg={bg_str}, delta={delta:.3f}, eventualBG={ev_str}, "
            f"prof_sens={prof_sens:.3f}, autosens_ratio={autosens_ratio:.3f}, effective_sens={effective_sens:.3f}, "
            f"insulinReq={ins_str}, duration={duration}, rate={rate:.3f}"
        )

        try:
            res.smb = smb
        except Exception:
            res.smb = None

        if trace_mode and tc is not None:
            return res, tc.dump()
        return res

    except Exception as e:
        auto_isf_consoleError.append(f"Exception in determine_basal_autoisf: {e}")
        if trace_mode and tc is not None:
            return res, tc.dump()
        return res
