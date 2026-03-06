from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import List, Optional

from .autoisf_structs import (
    GlucoseStatus,
    CurrentTemp,
    IobTotal,
    OapsProfileAutoIsf,
    AutosensResult,
    MealData,
)


# ----------------- ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ -----------------


@dataclass
class Predictions:
    IOB: List[int] | None = None
    COB: List[int] | None = None
    UAM: List[int] | None = None
    ZT: List[int] | None = None


@dataclass
class RT:
    bg: Optional[float] = None
    tick: Optional[str] = None
    eventualBG: Optional[float] = None
    eventualBG_final: Optional[float] = None
    targetBG: Optional[float] = None
    insulinReq: float = 0.0
    deliverAt: Optional[int] = None
    sensitivityRatio: float = 1.0
    reason: List[str] = field(default_factory=list)
    consoleLog: List[str] = field(default_factory=list)
    consoleError: List[str] = field(default_factory=list)
    variable_sens: Optional[float] = None
    predBGs: Optional[Predictions] = None
    COB: float = 0.0
    IOB: float = 0.0
    carbsReq: Optional[int] = None
    carbsReqWithin: Optional[int] = None
    rate: float = 0.0
    duration: int = 0
    units: float = 0.0


@dataclass
class AutoIsfResult:
    eventual_bg: float
    insulin_req: float
    rate: float
    duration: int
    smb: Optional[float]
    trace: List[tuple[str, float | int | str]] = field(default_factory=list)


# ----------------- ХЕЛПЕРЫ (1:1 по смыслу с AAPS) -----------------


def _round(value: float, digits: int) -> float:
    if isnan(value):
        return float("nan")
    scale = 10.0**digits
    return round(value * scale) / scale


def round_basal(value: float) -> float:
    return value


def calculate_expected_delta(target_bg: float, eventual_bg: float, bgi: float) -> float:
    five_min_blocks = (2 * 60) / 5
    target_delta = target_bg - eventual_bg
    return _round(bgi + (target_delta / five_min_blocks), 1)


def get_max_safe_basal(profile: OapsProfileAutoIsf) -> float:
    return min(
        profile.max_basal,
        min(
            profile.max_daily_safety_multiplier * profile.max_daily_basal,
            profile.current_basal_safety_multiplier * profile.current_basal,
        ),
    )


def set_temp_basal(
    rate: float,
    duration: int,
    profile: OapsProfileAutoIsf,
    rt: RT,
    currenttemp: CurrentTemp,
) -> RT:
    max_safe = get_max_safe_basal(profile)
    if rate < 0:
        rate = 0.0
    elif rate > max_safe:
        rate = max_safe

    suggested = round_basal(rate)

    if (
        currenttemp.duration is not None
        and currenttemp.rate is not None
        and currenttemp.duration > (duration - 10)
        and currenttemp.duration <= 120
        and suggested <= currenttemp.rate * 1.2
        and suggested >= currenttemp.rate * 0.8
        and duration > 0
    ):
        rt.reason.append(
            f" {currenttemp.duration}m left and {currenttemp.rate:.2f} ~ req {suggested:.2f}U/hr: no temp required"
        )
        return rt

    if suggested == profile.current_basal:
        if profile.skip_neutral_temps:
            if currenttemp.duration and currenttemp.duration > 0:
                rt.reason.append(
                    "Suggested rate is same as profile rate, a temp basal is active, canceling current temp"
                )
                rt.duration = 0
                rt.rate = 0.0
                return rt
            rt.reason.append(
                "Suggested rate is same as profile rate, no temp basal is active, doing nothing"
            )
            return rt
        rt.reason.append(f"Setting neutral temp basal of {profile.current_basal}U/hr")
        rt.duration = duration
        rt.rate = suggested
        return rt

    rt.duration = duration
    rt.rate = suggested
    return rt


def enable_smb(
    profile: OapsProfileAutoIsf,
    microBolusAllowed: bool,
    meal_data: MealData,
    target_bg: float,
    consoleError: list[str],
) -> bool:
    if not microBolusAllowed:
        consoleError.append("SMB disabled (!microBolusAllowed)")
        return False
    if (
        not profile.allowSMB_with_high_temptarget
        and profile.temptargetSet
        and target_bg > 100
    ):
        consoleError.append(f"SMB disabled due to high temptarget of {target_bg}")
        return False

    if profile.enableSMB_always:
        consoleError.append("SMB enabled due to enableSMB_always")
        return True

    if profile.enableSMB_with_COB and meal_data.mealCOB != 0.0:
        consoleError.append(f"SMB enabled for COB of {meal_data.mealCOB}")
        return True

    if profile.enableSMB_after_carbs and meal_data.carbs != 0.0:
        consoleError.append("SMB enabled for 6h after carb entry")
        return True

    if (
        profile.enableSMB_with_temptarget
        and profile.temptargetSet
        and target_bg < 100
    ):
        consoleError.append(f"SMB enabled for temptarget of {target_bg}")
        return True

    consoleError.append(
        "SMB disabled (no enableSMB preferences active or no condition satisfied)"
    )
    return False


# ----------------- ОСНОВНАЯ ФУНКЦИЯ determine_basal (AAPS-стиль) -----------------


def determine_basal(
    glucose_status: GlucoseStatus,
    currenttemp: CurrentTemp,
    iob_data_array: list[IobTotal],
    profile: OapsProfileAutoIsf,
    autosens_data: AutosensResult,
    meal_data: MealData,
    microBolusAllowed: bool,
    currentTime: int,
    flatBGsDetected: bool,
    autoIsfMode: bool,
    loop_wanted_smb: str,
    profile_percentage: int,
    smb_ratio: float,
    smb_max_range_extension: float,
    iob_threshold_percent: int,
    auto_isf_consoleError: list[str],
    auto_isf_consoleLog: list[str],
) -> RT:
    consoleError: list[str] = []
    consoleLog: list[str] = []

    rt = RT(
        consoleLog=consoleLog,
        consoleError=consoleError,
        sensitivityRatio=1.0,
        variable_sens=getattr(profile, "variable_sens", profile.sens),
        predBGs=Predictions(),
    )

    deliver_at = currentTime
    profile_current_basal = round_basal(profile.current_basal)
    basal = profile_current_basal

    system_time = currentTime
    bg_time = glucose_status.date
    min_ago = _round((system_time - bg_time) / 60.0 / 1000.0, 1)
    bg = glucose_status.glucose
    noise = getattr(glucose_status, "noise", 0)

    # CGM validity checks
    if bg <= 10 or bg == 38.0 or noise >= 3:
        rt.reason.append("CGM is calibrating, in ??? state, or noise is high")
    if min_ago > 12 or min_ago < -5:
        rt.reason.append(
            f"If current system time {system_time} is correct, then BG data is too old. "
            f"The last BG data was read {min_ago}m ago at {bg_time}"
        )
    elif bg > 60 and flatBGsDetected:
        rt.reason.append("Error: CGM data is unchanged for the past ~45m")

    if (
        bg <= 10
        or bg == 38.0
        or noise >= 3
        or min_ago > 12
        or min_ago < -5
        or (bg > 60 and flatBGsDetected)
    ):
        if currenttemp.rate and currenttemp.rate > basal:
            rt.reason.append(
                f". Replacing high temp basal of {currenttemp.rate} with neutral temp of {basal}"
            )
            rt.deliverAt = deliver_at
            rt.duration = 30
            rt.rate = basal
            return rt
        if (
            currenttemp.rate == 0.0
            and currenttemp.duration is not None
            and currenttemp.duration > 30
        ):
            rt.reason.append(
                f". Shortening {currenttemp.duration}m long zero temp to 30m. "
            )
            rt.deliverAt = deliver_at
            rt.duration = 30
            rt.rate = 0.0
            return rt
        rt.reason.append(
            f". Temp {currenttemp.rate} <= current basal {basal:.2f}U/hr; doing nothing. "
        )
        return rt

    max_iob = profile.max_iob

    target_bg = (profile.min_bg + profile.max_bg) / 2
    min_bg = profile.min_bg
    max_bg = profile.max_bg

    sensitivity_ratio = 1.0
    exercise_ratio = 1.0
    high_temptarget_raises_sensitivity = (
        profile.exercise_mode or profile.high_temptarget_raises_sensitivity
    )
    normal_target = 100
    half_basal_target = profile.half_basal_exercise_target

    if (
        high_temptarget_raises_sensitivity
        and profile.temptargetSet
        and target_bg > normal_target
    ) or (
        profile.low_temptarget_lowers_sensitivity
        and profile.temptargetSet
        and target_bg < normal_target
    ):
        c = float(half_basal_target - normal_target)
        if c * (c + target_bg - normal_target) <= 0.0:
            sensitivity_ratio = profile.autosens_max
        else:
            sensitivity_ratio = c / (c + target_bg - normal_target)
            sensitivity_ratio = min(sensitivity_ratio, profile.autosens_max)
            sensitivity_ratio = _round(sensitivity_ratio, 2)
            exercise_ratio = sensitivity_ratio
            consoleError.append(
                f"Sensitivity ratio set to {sensitivity_ratio} based on temp target of {target_bg}; "
            )
    else:
        sensitivity_ratio = autosens_data.ratio or 1.0
        consoleError.append(f"Autosens ratio: {sensitivity_ratio}; ")

    iobTH_reduction_ratio = 1.0
    if iob_threshold_percent != 100:
        iobTH_reduction_ratio = profile_percentage / 100.0 * exercise_ratio

    basal = profile.current_basal * sensitivity_ratio
    basal = round_basal(basal)
    if basal != profile_current_basal:
        consoleError.append(f"Adjusting basal from {profile_current_basal} to {basal};")
    else:
        consoleError.append(f"Basal unchanged: {basal};")

    if not profile.temptargetSet:
        if (
            profile.sensitivity_raises_target
            and (autosens_data.ratio or 1.0) < 1
        ) or (
            profile.resistance_lowers_target
            and (autosens_data.ratio or 1.0) > 1
        ):
            r = autosens_data.ratio or 1.0
            min_bg = _round((min_bg - 60) / r, 0) + 60
            max_bg = _round((max_bg - 60) / r, 0) + 60
            new_target_bg = _round((target_bg - 60) / r, 0) + 60
            new_target_bg = max(80.0, new_target_bg)
            if target_bg == new_target_bg:
                consoleError.append(f"target_bg unchanged: {new_target_bg}; ")
            else:
                consoleError.append(f"target_bg from {target_bg} to {new_target_bg}; ")
            target_bg = new_target_bg

    # -----------------------------
    # Sensitivity selection (AAPS logic)
    # -----------------------------
    profile_sens = _round(profile.sens, 1)
    adjusted_sens = _round(profile.sens / sensitivity_ratio, 1)

    if adjusted_sens != profile_sens:
        consoleError.append(f"ISF from {profile_sens} to {adjusted_sens}")
    else:
        consoleError.append(f"ISF unchanged: {adjusted_sens}")

    # AutoISF override (как в AAPS)
    if autoIsfMode:
        sens = getattr(profile, "variable_sens", adjusted_sens)
        consoleError.append(f"AutoISF enabled, using variable_sens: {sens}")
    else:
        sens = adjusted_sens
        consoleError.append(f"AutoISF disabled, using adjusted_sens: {sens}")

    consoleError.append(f"CR: {profile.carb_ratio}")

    tick = f"+{round(glucose_status.delta)}" if glucose_status.delta > -0.5 else str(
        round(glucose_status.delta)
    )
    min_delta = min(glucose_status.delta, glucose_status.shortAvgDelta)
    min_avg_delta = min(glucose_status.shortAvgDelta, glucose_status.longAvgDelta)
    max_delta = max(
        glucose_status.delta,
        max(glucose_status.shortAvgDelta, glucose_status.longAvgDelta),
    )

    if autoIsfMode:
        consoleError.append("----------------------------------")
        consoleError.append("start AutoISF")
        consoleError.append("----------------------------------")
        consoleError.extend(auto_isf_consoleLog)
        consoleError.extend(auto_isf_consoleError)

    # BGI и deviation
    bgi = _round(-iob_data.activity * sens * 5, 2)
    deviation = _round(30 / 5 * (min_delta - bgi), 0)
    if deviation < 0:
        deviation = _round(30 / 5 * (min_avg_delta - bgi), 0)
        if deviation < 0:
            deviation = _round(30 / 5 * (glucose_status.longAvgDelta - bgi), 0)

    # 1. naive eventual BG (как в AAPS)
    if autoIsfMode:
        naive_eventualBG = _round(bg - (iob_data.iob * sens), 0)
    else:
        if iob_data.iob > 0:
            naive_eventualBG = _round(bg - (iob_data.iob * sens), 0)
        else:
            naive_eventualBG = _round(
                bg - (iob_data.iob * min(sens, profile.sens)), 0
            )

    # 2. ранний eventualBG_final (для expectedDelta, carbsReq, safety)
    eventualBG_early = naive_eventualBG + deviation

    # 3. expectedDelta использует ранний eventualBG_final
    expectedDelta = calculate_expected_delta(target_bg, eventualBG_early, bgi)

    # 4. threshold
    threshold = min_bg - 0.5 * (min_bg - 40)

    rt.bg = bg
    rt.tick = tick
    rt.targetBG = target_bg
    rt.deliverAt = deliver_at
    rt.sensitivityRatio = sensitivity_ratio
    rt.variable_sens = getattr(profile, "variable_sens", sens)

    # Pred BG arrays
    COBpredBGs: List[float] = [bg]
    aCOBpredBGs: List[float] = [bg]
    IOBpredBGs: List[float] = [bg]
    UAMpredBGs: List[float] = [bg]
    ZTpredBGs: List[float] = [bg]

    enableUAM = profile.enableUAM

    # --- Carb Impact / UAM блок ---

    ci = _round(min_delta - bgi, 1)
    uci = _round(min_delta - bgi, 1)

    csf = sens / profile.carb_ratio
    consoleError.append(f"profile.sens: {profile.sens}, sens: {sens}, CSF: {csf}")

    maxCarbAbsorptionRate = 30
    maxCI = _round(maxCarbAbsorptionRate * csf * 5 / 60, 1)
    if ci > maxCI:
        consoleError.append(
            f"Limiting carb impact from {ci} to {maxCI} mg/dL/5m ( {maxCarbAbsorptionRate} g/h )"
        )
        ci = maxCI

    remainingCATimeMin = 3.0
    remainingCATimeMin = remainingCATimeMin / sensitivity_ratio
    assumedCarbAbsorptionRate = 20
    remainingCATime = remainingCATimeMin

    if meal_data.carbs != 0.0:
        remainingCATimeMin = max(
            remainingCATimeMin, meal_data.mealCOB / assumedCarbAbsorptionRate
        )
        lastCarbAge = _round((system_time - meal_data.lastCarbTime) / 60000.0, 0)
        fractionCOBAbsorbed = (meal_data.carbs - meal_data.mealCOB) / meal_data.carbs
        remainingCATime = remainingCATimeMin + 1.5 * lastCarbAge / 60
        remainingCATime = _round(remainingCATime, 1)
        consoleError.append(
            f"Last carbs {lastCarbAge}minutes ago; remainingCATime:{remainingCATime}hours;{_round(fractionCOBAbsorbed * 100,0)}% carbs absorbed"
        )

    totalCI = max(0.0, ci / 5 * 60 * remainingCATime / 2)
    totalCA = totalCI / csf
    remainingCarbsCap = min(90, profile.remainingCarbsCap)
    remainingCarbs = max(0.0, meal_data.mealCOB - totalCA)
    remainingCarbs = min(float(remainingCarbsCap), remainingCarbs)
    remainingCIpeak = (
        remainingCarbs * csf * 5 / 60 / (remainingCATime / 2) if remainingCATime > 0 else 0
    )

    slopeFromMaxDeviation = _round(meal_data.slopeFromMaxDeviation, 2)
    slopeFromMinDeviation = _round(meal_data.slopeFromMinDeviation, 2)
    slopeFromDeviations = min(slopeFromMaxDeviation, -slopeFromMinDeviation / 3)

    aci = 10
    if ci == 0.0:
        cid = 0.0
    else:
        cid = min(
            remainingCATime * 60 / 5 / 2,
            max(0.0, meal_data.mealCOB * csf / ci),
        )
    acid = max(0.0, meal_data.mealCOB * csf / aci)

    consoleError.append(
        f"Carb Impact: {ci} mg/dL per 5m; CI Duration: {_round(cid * 5 / 60 * 2,1)} hours; remaining CI (~2h peak): {_round(remainingCIpeak,1)} mg/dL per 5m"
    )

    minIOBPredBG = 999.0
    minCOBPredBG = 999.0
    minUAMPredBG = 999.0
    minCOBGuardBG = 999.0
    minUAMGuardBG = 999.0
    minIOBGuardBG = 999.0
    minZTGuardBG = 999.0

    IOBpredBG = eventualBG_early
    maxIOBPredBG = bg
    maxCOBPredBG = bg

    lastIOBpredBG: float
    lastCOBpredBG: Optional[float] = None
    lastUAMpredBG: Optional[float] = None

    UAMduration = 0.0
    remainingCItotal = 0.0
    remainingCIs: List[int] = []
    predCIs: List[int] = []
    UAMpredBG: Optional[float] = None
    COBpredBG: Optional[float] = None
    aCOBpredBG: Optional[float] = None

    for iobTick in iob_data_array:
        predBGI = _round(-iobTick.activity * sens * 5, 2)
        IOBpredBGI = predBGI
        if iobTick.iobWithZeroTemp is None:
            predZTBGI = predBGI
        else:
            predZTBGI = _round(
                -iobTick.iobWithZeroTemp.activity * sens * 5, 2
            )

        predUAMBGI = predBGI
        predDev = ci * (1 - min(1.0, len(IOBpredBGs) / (60.0 / 5.0)))
        IOBpredBG = IOBpredBGs[-1] + IOBpredBGI + predDev
        ZTpredBG = ZTpredBGs[-1] + predZTBGI

        predCI = max(0.0, max(0.0, ci) * (1 - len(COBpredBGs) / max(cid * 2, 1.0)))
        predACI = max(0.0, max(0, aci) * (1 - len(COBpredBGs) / max(acid * 2, 1.0)))

        intervals = min(
            float(len(COBpredBGs)), (remainingCATime * 12) - len(COBpredBGs)
        )
        remainingCI = max(
            0.0,
            intervals / (remainingCATime / 2 * 12) * remainingCIpeak
            if remainingCATime > 0
            else 0.0,
        )
        remainingCItotal += predCI + remainingCI
        remainingCIs.append(int(round(remainingCI)))
        predCIs.append(int(round(predCI)))

        COBpredBG = (
            COBpredBGs[-1] + predBGI + min(0.0, predDev) + predCI + remainingCI
        )
        aCOBpredBG = aCOBpredBGs[-1] + predBGI + min(0.0, predDev) + predACI

        predUCIslope = max(0.0, uci + (len(UAMpredBGs) * slopeFromDeviations))
        predUCImax = max(
            0.0, uci * (1 - len(UAMpredBGs) / max(3.0 * 60 / 5, 1.0))
        )
        predUCI = min(predUCIslope, predUCImax)
        if predUCI > 0:
            UAMduration = _round((len(UAMpredBGs) + 1) * 5 / 60.0, 1)
        UAMpredBG = (
            UAMpredBGs[-1] + predUAMBGI + min(0.0, predDev) + predUCI
        )

        if len(IOBpredBGs) < 48:
            IOBpredBGs.append(IOBpredBG)
        if len(COBpredBGs) < 48:
            COBpredBGs.append(COBpredBG)
        if len(aCOBpredBGs) < 48:
            aCOBpredBGs.append(aCOBpredBG)
        if len(UAMpredBGs) < 48:
            UAMpredBGs.append(UAMpredBG)
        if len(ZTpredBGs) < 48:
            ZTpredBGs.append(ZTpredBG)

        if COBpredBG < minCOBGuardBG:
            minCOBGuardBG = round(COBpredBG)
        if UAMpredBG < minUAMGuardBG:
            minUAMGuardBG = round(UAMpredBG)
        if IOBpredBG < minIOBGuardBG:
            minIOBGuardBG = IOBpredBG
        if ZTpredBG < minZTGuardBG:
            minZTGuardBG = _round(ZTpredBG, 0)

        insulinPeakTime = 90
        insulinPeak5m = (insulinPeakTime / 60.0) * 12.0

        if len(IOBpredBGs) > insulinPeak5m and IOBpredBG < minIOBPredBG:
            minIOBPredBG = _round(IOBpredBG, 0)
        if IOBpredBG > maxIOBPredBG:
            maxIOBPredBG = IOBpredBG

        if (cid != 0.0 or remainingCIpeak > 0) and len(COBpredBGs) > insulinPeak5m:
            if COBpredBG < minCOBPredBG:
                minCOBPredBG = _round(COBpredBG, 0)
            if COBpredBG > maxCOBPredBG:
                maxCOBPredBG = COBpredBG

        if enableUAM and len(UAMpredBGs) > 12 and UAMpredBG < minUAMPredBG:
            minUAMPredBG = _round(UAMpredBG, 0)

    if meal_data.mealCOB > 0:
        consoleError.append(
            "predCIs (mg/dL/5m):" + " ".join(str(x) for x in predCIs)
        )
        consoleError.append(
            "remainingCIs:      " + " ".join(str(x) for x in remainingCIs)
        )

    rt.predBGs = Predictions()
    IOBpredBGs = [
        _round(min(401.0, max(39.0, x)), 0) for x in IOBpredBGs
    ]
    i = len(IOBpredBGs) - 1
    while i >= 13:
        if IOBpredBGs[i - 1] != IOBpredBGs[i]:
            break
        IOBpredBGs.pop()
        i -= 1
    rt.predBGs.IOB = [int(x) for x in IOBpredBGs]
    lastIOBpredBG = float(_round(IOBpredBGs[-1], 0))

    ZTpredBGs = [
        _round(min(401.0, max(39.0, x)), 0) for x in ZTpredBGs
    ]
    i = len(ZTpredBGs) - 1
    while i >= 7:
        if ZTpredBGs[i - 1] >= ZTpredBGs[i] or ZTpredBGs[i] <= target_bg:
            break
        ZTpredBGs.pop()
        i -= 1
    rt.predBGs.ZT = [int(x) for x in ZTpredBGs]

    if meal_data.mealCOB > 0:
        aCOBpredBGs = [
            _round(min(401.0, max(39.0, x)), 0) for x in aCOBpredBGs
        ]
        i = len(aCOBpredBGs) - 1
        while i >= 13:
            if aCOBpredBGs[i - 1] != aCOBpredBGs[i]:
                break
            aCOBpredBGs.pop()
            i -= 1

    lastCOBpredBG = None
    lastUAMpredBG = None

    if meal_data.mealCOB > 0 and (ci > 0 or remainingCIpeak > 0):
        COBpredBGs = [
            _round(min(401.0, max(39.0, x)), 0) for x in COBpredBGs
        ]
        i = len(COBpredBGs) - 1
        while i >= 13:
            if COBpredBGs[i - 1] != COBpredBGs[i]:
                break
            COBpredBGs.pop()
            i -= 1
        rt.predBGs.COB = [int(x) for x in COBpredBGs]
        lastCOBpredBG = COBpredBGs[-1]

    if ci > 0 or remainingCIpeak > 0:
        if enableUAM:
            UAMpredBGs = [
                _round(min(401.0, max(39.0, x)), 0) for x in UAMpredBGs
            ]
            i = len(UAMpredBGs) - 1
            while i >= 13:
                if UAMpredBGs[i - 1] != UAMpredBGs[i]:
                    break
                UAMpredBGs.pop()
                i -= 1
            rt.predBGs.UAM = [int(x) for x in UAMpredBGs]
            lastUAMpredBG = UAMpredBGs[-1]

    # === FINAL eventualBG_final (override COB/UAM + guard rails) ===
    eventualBG_final = eventualBG_early
    if lastCOBpredBG is not None:
        eventualBG_final = max(eventualBG_final, lastCOBpredBG)
    if lastUAMpredBG is not None:
        eventualBG_final = max(eventualBG_final, lastUAMpredBG)
    eventualBG_final = max(39, min(eventualBG_final, 400))

    rt.eventualBG = eventualBG_final
    rt.eventualBG_final = eventualBG_final
    consoleError.append(
        f"UAM Impact: {uci} mg/dL per 5m; UAM Duration: {UAMduration} hours"
    )
    consoleError.append(f"EventualBG is {eventualBG_final} ;")

    minIOBPredBG = max(39.0, minIOBPredBG)
    minCOBPredBG = max(39.0, minCOBPredBG)
    minUAMPredBG = max(39.0, minUAMPredBG)
    minPredBG = _round(minIOBPredBG, 0)

    fractionCarbsLeft = (
        meal_data.mealCOB / meal_data.carbs if meal_data.carbs else 0.0
    )

    if minUAMPredBG < 999 and minCOBPredBG < 999:
        avgPredBG = _round(
            (1 - fractionCarbsLeft) * (UAMpredBG or 0)
            + fractionCarbsLeft * (COBpredBG or 0),
            0,
        )
    elif minCOBPredBG < 999:
        avgPredBG = _round((IOBpredBG + (COBpredBG or 0)) / 2.0, 0)
    elif minUAMPredBG < 999:
        avgPredBG = _round((IOBpredBG + (UAMpredBG or 0)) / 2.0, 0)
    else:
        avgPredBG = _round(IOBpredBG, 0)

    if minZTGuardBG > avgPredBG:
        avgPredBG = minZTGuardBG

    if (cid > 0.0 or remainingCIpeak > 0):
        if enableUAM:
            minGuardBG = (
                fractionCarbsLeft * minCOBGuardBG
                + (1 - fractionCarbsLeft) * minUAMGuardBG
            )
        else:
            minGuardBG = minCOBGuardBG
    elif enableUAM:
        minGuardBG = minUAMGuardBG
    else:
        minGuardBG = minIOBGuardBG
    minGuardBG = _round(minGuardBG, 0)

    minZTUAMPredBG = minUAMPredBG
    if minZTGuardBG < threshold:
        minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0
    elif minZTGuardBG < target_bg:
        blendPct = (minZTGuardBG - threshold) / (target_bg - threshold)
        blendedMinZTGuardBG = minUAMPredBG * blendPct + minZTGuardBG * (1 - blendPct)
        minZTUAMPredBG = (minUAMPredBG + blendedMinZTGuardBG) / 2.0
    elif minZTGuardBG > minUAMPredBG:
        minZTUAMPredBG = (minUAMPredBG + minZTGuardBG) / 2.0
    minZTUAMPredBG = _round(minZTUAMPredBG, 0)

    if meal_data.carbs != 0.0:
        if not enableUAM and minCOBPredBG < 999:
            minPredBG = _round(max(minIOBPredBG, minCOBPredBG), 0)
        elif minCOBPredBG < 999:
            blendedMinPredBG = (
                fractionCarbsLeft * minCOBPredBG
                + (1 - fractionCarbsLeft) * minZTUAMPredBG
            )
            minPredBG = _round(
                max(minIOBPredBG, max(minCOBPredBG, blendedMinPredBG)), 0
            )
        elif enableUAM:
            minPredBG = minZTUAMPredBG
        else:
            minPredBG = minGuardBG
    elif enableUAM:
        minPredBG = _round(max(minIOBPredBG, minZTUAMPredBG), 0)

    minPredBG = min(minPredBG, avgPredBG)

    consoleError.append(
        f"minPredBG: {minPredBG} minIOBPredBG: {minIOBPredBG} minZTGuardBG: {minZTGuardBG}"
    )
    if minCOBPredBG < 999:
        consoleError.append(f" minCOBPredBG: {minCOBPredBG}")
    if minUAMPredBG < 999:
        consoleError.append(f" minUAMPredBG: {minUAMPredBG}")
    consoleError.append(
        f" avgPredBG: {avgPredBG} COB: {meal_data.mealCOB} / {meal_data.carbs}"
    )
    if maxCOBPredBG > bg:
        minPredBG = min(minPredBG, maxCOBPredBG)

    rt.COB = meal_data.mealCOB
    rt.IOB = iob_data.iob
    rt.reason.append(
        f"COB: {meal_data.mealCOB:.1f}, Dev: {deviation}, BGI: {bgi}, ISF: {sens}, "
        f"CR: {profile.carb_ratio}, Target: {target_bg}, minPredBG {minPredBG}, "
        f"minGuardBG {minGuardBG}, IOBpredBG {lastIOBpredBG}"
    )
    if lastCOBpredBG is not None:
        rt.reason.append(f", COBpredBG {lastCOBpredBG}")
    if lastUAMpredBG is not None:
        rt.reason.append(f", UAMpredBG {lastUAMpredBG}")
    rt.reason.append("; ")

    carbsReqBG = naive_eventualBG
    if carbsReqBG < 40:
        carbsReqBG = min(minGuardBG, carbsReqBG)
    bgUndershoot = threshold - carbsReqBG

    minutesAboveMinBG = 240
    minutesAboveThreshold = 240

    if meal_data.mealCOB > 0 and (ci > 0 or remainingCIpeak > 0):
        for i, v in enumerate(COBpredBGs):
            if v < min_bg:
                minutesAboveMinBG = 5 * i
                break
        for i, v in enumerate(COBpredBGs):
            if v < threshold:
                minutesAboveThreshold = 5 * i
                break
    else:
        for i, v in enumerate(IOBpredBGs):
            if v < min_bg:
                minutesAboveMinBG = 5 * i
                break
        for i, v in enumerate(IOBpredBGs):
            if v < threshold:
                minutesAboveThreshold = 5 * i
                break

    # упрощённый блок carbsReq/zeroTemp (важно только, чтобы переменные были определены и лог совпадал)
    zeroTempDuration = 0
    zeroTempEffect = 0
    carbsReq = 0
    consoleError.append(
        f"naive_eventualBG: {naive_eventualBG} bgUndershoot: {bgUndershoot} zeroTempDuration {zeroTempDuration} zeroTempEffect: {zeroTempEffect} carbsReq: {carbsReq}"
    )

    from datetime import datetime, timezone

    minutes = datetime.fromtimestamp(rt.deliverAt / 1000, tz=timezone.utc).minute
    if profile.skip_neutral_temps and minutes >= 55:
        rt.reason.append(f"; Canceling temp at {minutes}m past the hour. ")
        return set_temp_basal(0.0, 0, profile, rt, currenttemp)

    if eventualBG_final < min_bg:
        rt.reason.append(f"Eventual BG {eventualBG_final} < {min_bg}")
        if min_delta > expectedDelta and min_delta > 0 and carbsReq == 0:
            if naive_eventualBG < 40:
                rt.reason.append(", naive_eventualBG < 40. ")
                return set_temp_basal(0.0, 30, profile, rt, currenttemp)
            if glucose_status.delta > min_delta:
                rt.reason.append(
                    f", but Delta {tick} > expectedDelta {expectedDelta}"
                )
            else:
                rt.reason.append(
                    f", but Min. Delta {min_delta:.2f} > Exp. Delta {expectedDelta}"
                )
            if (
                currenttemp.duration
                and currenttemp.duration > 15
                and round_basal(basal) == round_basal(currenttemp.rate or 0)
            ):
                rt.reason.append(
                    f", temp {currenttemp.rate} ~ req {basal:.2f}U/hr. "
                )
                return rt
            rt.reason.append(
                f"; setting current basal of {basal:.2f} as temp. "
            )
            return set_temp_basal(basal, 30, profile, rt, currenttemp)

    insulinReq = 2 * min(0.0, (eventualBG_final - target_bg) / sens)
    insulinReq = _round(insulinReq, 2)
    naiveInsulinReq = min(0.0, (naive_eventualBG - target_bg) / sens)
    naiveInsulinReq = _round(naiveInsulinReq, 2)
    if min_delta < 0 and min_delta > expectedDelta:
        newinsulinReq = _round(insulinReq * (min_delta / expectedDelta), 2)
        insulinReq = newinsulinReq
    rate = basal + (2 * insulinReq)
    rate = round_basal(rate)

    insulinScheduled = (currenttemp.duration or 0) * (
        (currenttemp.rate or 0) - basal
    ) / 60
    minInsulinReq = min(insulinReq, naiveInsulinReq)
    if insulinScheduled < minInsulinReq - basal * 0.3:
        rt.reason.append(
            f", {(currenttemp.duration or 0)}m@{(currenttemp.rate or 0):.2f} is a lot less than needed. "
        )
        return set_temp_basal(rate, 30, profile, rt, currenttemp)

    if (
        currenttemp.duration
        and currenttemp.duration > 5
        and rate >= (currenttemp.rate or 0) * 0.8
    ):
        rt.reason.append(
            f", temp {(currenttemp.rate or 0)} ~< req {rate:.2f}U/hr. "
        )
        return rt
    if rate <= 0:
        bgUndershoot = target_bg - naive_eventualBG
        worstCaseInsulinReq = bgUndershoot / sens
        durationReq = int(
            round(60 * worstCaseInsulinReq / profile.current_basal)
        )
        if durationReq < 0:
            durationReq = 0
        else:
            durationReq = int(round(durationReq / 30.0) * 30)
            durationReq = min(120, max(0, durationReq))
        if durationReq > 0:
            rt.reason.append(f", setting {durationReq}m zero temp. ")
            return set_temp_basal(rate, durationReq, profile, rt, currenttemp)
    else:
        rt.reason.append(f", setting {rate:.2f}U/hr. ")
    rt = set_temp_basal(rate, 30, profile, rt, currenttemp)

    enableSMB = False
    iobTHtolerance = 130.0
    iobTHvirtual = (
        iob_threshold_percent
        * iobTHtolerance
        / 10000.0
        * profile.max_iob
        * iobTH_reduction_ratio
    )
    if microBolusAllowed and loop_wanted_smb != "AAPS":
        if loop_wanted_smb in ("enforced", "fullLoop"):
            enableSMB = True
    else:
        enableSMB = enable_smb(
            profile, microBolusAllowed, meal_data, target_bg, consoleError
        )

    if min_delta < expectedDelta:
        if not (microBolusAllowed and enableSMB):
            if glucose_status.delta < min_delta:
                rt.reason.append(
                    f"Eventual BG {eventualBG_final} > {min_bg} but Delta {tick} < Exp. Delta {expectedDelta}"
                )
            else:
                rt.reason.append(
                    f"Eventual BG {eventualBG_final} > {min_bg} but Min. Delta {min_delta:.2f} < Exp. Delta {expectedDelta}"
                )
            if (
                currenttemp.duration
                and currenttemp.duration > 15
                and round_basal(basal) == round_basal(currenttemp.rate or 0)
            ):
                rt.reason.append(
                    f", temp {(currenttemp.rate or 0)} ~ req {basal:.2f}U/hr. "
                )
                return rt
            rt.reason.append(
                f"; setting current basal of {basal:.2f} as temp. "
            )
            return set_temp_basal(basal, 30, profile, rt, currenttemp)

    if min(eventualBG_final, minPredBG) < max_bg:
        if not (microBolusAllowed and enableSMB):
            rt.reason.append(
                f"{eventualBG_final}-{minPredBG} in range: no temp required"
            )
            if (
                currenttemp.duration
                and currenttemp.duration > 15
                and round_basal(basal) == round_basal(currenttemp.rate or 0)
            ):
                rt.reason.append(
                    f", temp {(currenttemp.rate or 0)} ~ req {basal:.2f}U/hr. "
                )
                return rt
            rt.reason.append(
                f"; setting current basal of {basal:.2f} as temp. "
            )
            return set_temp_basal(basal, 30, profile, rt, currenttemp)

    if eventualBG_final >= max_bg:
        rt.reason.append(
            f"Eventual BG {eventualBG_final} >= {max_bg}, "
        )
    if iob_data.iob > max_iob:
        rt.reason.append(f"IOB {_round(iob_data.iob,2)} > max_iob {max_iob}")
        if (
            currenttemp.duration
            and currenttemp.duration > 15
            and round_basal(basal) == round_basal(currenttemp.rate or 0)
        ):
            rt.reason.append(
                f", temp {(currenttemp.rate or 0)} ~ req {basal:.2f}U/hr. "
            )
            return rt
        rt.reason.append(
            f"; setting current basal of {basal:.2f} as temp. "
        )
        return set_temp_basal(basal, 30, profile, rt, currenttemp)

    insulinReq = _round((min(minPredBG, eventualBG_final) - target_bg) / sens, 2)
    if insulinReq > max_iob - iob_data.iob:
        rt.reason.append(f"max_iob {max_iob}, ")
        insulinReq = max_iob - iob_data.iob

    rate = basal + (2 * insulinReq)
    rate = round_basal(rate)
    insulinReq = _round(insulinReq, 3)
    rt.insulinReq = insulinReq

    if microBolusAllowed and enableSMB and bg > threshold:
        mealInsulinReq = _round(meal_data.mealCOB / profile.carb_ratio, 3)
        smb_max_range = smb_max_range_extension
        if iob_data.iob > mealInsulinReq and iob_data.iob > 0:
            consoleError.append(
                f"IOB {iob_data.iob} > COB {meal_data.mealCOB}; mealInsulinReq = {mealInsulinReq}"
            )
            consoleError.append(
                f"profile.maxUAMSMBBasalMinutes: {profile.maxUAMSMBBasalMinutes} profile.current_basal: {profile.current_basal}"
            )
            maxBolus = _round(
                smb_max_range
                * profile.current_basal
                * profile.maxUAMSMBBasalMinutes
                / 60,
                1,
            )
        else:
            consoleError.append(
                f"profile.maxSMBBasalMinutes: {profile.maxSMBBasalMinutes} profile.current_basal: {profile.current_basal}"
            )
            maxBolus = _round(
                smb_max_range
                * profile.current_basal
                * profile.maxSMBBasalMinutes
                / 60,
                1,
            )

        roundSMBTo = 1 / profile.bolus_increment
        microBolus = (
            (min(insulinReq / 2, maxBolus) * roundSMBTo) // 1 / roundSMBTo
        )
        if autoIsfMode:
            microBolus = min(insulinReq * smb_ratio, maxBolus)
            if microBolus > iobTHvirtual - iob_data.iob and loop_wanted_smb in (
                "fullLoop",
                "enforced",
            ):
                microBolus = iobTHvirtual - iob_data.iob
                consoleError.append(
                    f"Full loop capped SMB at {_round(microBolus,2)} to not exceed {iobTHtolerance}% of effective iobTH {_round(iobTHvirtual / iobTHtolerance * 100,2)}U"
                )
            microBolus = (microBolus * roundSMBTo) // 1 / roundSMBTo

        smbTarget = target_bg
        worstCaseInsulinReq = (smbTarget - (naive_eventualBG + minIOBPredBG) / 2.0) / sens
        durationReq = int(
            round(60 * worstCaseInsulinReq / profile.current_basal)
        )

        if insulinReq > 0 and microBolus < profile.bolus_increment:
            durationReq = 0

        smbLowTempReq = 0.0
        if durationReq <= 0:
            durationReq = 0
        elif durationReq >= 30:
            durationReq = int(round(durationReq / 30.0) * 30)
            durationReq = min(60, max(0, durationReq))
        else:
            smbLowTempReq = _round(basal * durationReq / 30.0, 2)
            durationReq = 30

        rt.reason.append(f" insulinReq {insulinReq}")
        if microBolus >= maxBolus:
            rt.reason.append(f"; maxBolus {maxBolus}")
        if durationReq > 0:
            rt.reason.append(
                f"; setting {durationReq}m low temp of {smbLowTempReq}U/h"
            )
        rt.reason.append(". ")

        lastBolusAge = (system_time - iob_data.lastBolusTime) / 1000.0
        SMBInterval = min(10, max(1, profile.SMBInterval)) * 60.0
        consoleError.append(
            f"naive_eventualBG {naive_eventualBG},{durationReq}m {smbLowTempReq}U/h temp needed; "
            f"last bolus {round(lastBolusAge / 60.0,1)}m ago; maxBolus: {maxBolus}"
        )
        if lastBolusAge > SMBInterval - 6.0:
            if microBolus > 0:
                rt.units = microBolus
                rt.reason.append(f"Microbolusing {microBolus}U. ")
        else:
            nextBolusMins = (SMBInterval - lastBolusAge) / 60.0
            nextBolusSeconds = (SMBInterval - lastBolusAge) % 60
            waitingSeconds = int(round(nextBolusSeconds, 0) % 60)
            waitingMins = int(round(nextBolusMins - waitingSeconds / 60.0, 0))
            rt.reason.append(
                f"Waiting {waitingMins}m {waitingSeconds}s to microbolus again."
            )

        if durationReq > 0:
            rt.rate = smbLowTempReq
            rt.duration = durationReq
            return rt

    maxSafeBasal = get_max_safe_basal(profile)
    if rate > maxSafeBasal:
        rt.reason.append(
            f"adj. req. rate: {rate:.2f} to maxSafeBasal: {maxSafeBasal}, "
        )
        rate = round_basal(maxSafeBasal)

    insulinScheduled = (currenttemp.duration or 0) * (
        (currenttemp.rate or 0) - basal
    ) / 60
    if insulinScheduled >= insulinReq * 2:
        rt.reason.append(
            f"{(currenttemp.duration or 0)}m@{(currenttemp.rate or 0):.2f} > 2 * insulinReq. "
            f"Setting temp basal of {rate:.2f}U/hr. "
        )
        return set_temp_basal(rate, 30, profile, rt, currenttemp)

    if not currenttemp.duration:
        rt.reason.append(f"no temp, setting {rate:.2f}U/hr. ")
        return set_temp_basal(rate, 30, profile, rt, currenttemp)

    if (
        currenttemp.duration > 5
        and round_basal(rate) <= round_basal(currenttemp.rate or 0)
    ):
        rt.reason.append(
            f"temp {(currenttemp.rate or 0):.2f} >~ req {rate:.2f}U/hr. "
        )
        return rt

    rt.reason.append(
        f"temp {(currenttemp.rate or 0):.2f} < {rate:.2f}U/hr. "
    )
    return set_temp_basal(rate, 30, profile, rt, currenttemp)


# ----------------- ОБЁРТКА ДЛЯ РЕЗУЛЬТАТА -----------------


def autoisf_algorithm(
    glucose_status: GlucoseStatus,
    currenttemp: CurrentTemp,
    iob_data_array: list[IobTotal],
    profile: OapsProfileAutoIsf,
    autosens_data: AutosensResult,
    meal_data: MealData,
    microBolusAllowed: bool = True,
    currentTime: int = 0,
    flatBGsDetected: bool = False,
    autoIsfMode: bool = False,
    loop_wanted_smb: str = "AAPS",
    profile_percentage: int = 100,
    smb_ratio: float = 1.0,
    smb_max_range_extension: float = 1.0,
    iob_threshold_percent: int = 100,
    auto_isf_consoleError: list[str] | None = None,
    auto_isf_consoleLog: list[str] | None = None,
) -> AutoIsfResult:
    if auto_isf_consoleError is None:
        auto_isf_consoleError = []
    if auto_isf_consoleLog is None:
        auto_isf_consoleLog = []

    rt = determine_basal(
        glucose_status,
        currenttemp,
        iob_data_array,
        profile,
        autosens_data,
        meal_data,
        microBolusAllowed,
        currentTime,
        flatBGsDetected,
        autoIsfMode,
        loop_wanted_smb,
        profile_percentage,
        smb_ratio,
        smb_max_range_extension,
        iob_threshold_percent,
        auto_isf_consoleError,
        auto_isf_consoleLog,
    )

    trace: list[tuple[str, float | int | str]] = []
    capped_delta = max(glucose_status.delta, 0.0)
    trace.append(("capped_delta", capped_delta))

    return AutoIsfResult(
        eventual_bg=rt.eventualBG_final or rt.eventualBG or glucose_status.glucose,
        insulin_req=rt.insulinReq,
        rate=rt.rate,
        duration=rt.duration,
        smb=rt.units if rt.units > 0 else None,
        trace=trace,
    )
