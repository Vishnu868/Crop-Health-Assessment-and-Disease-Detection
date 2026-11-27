"""Smart-sensing irrigation logic (mirrors firmware/nodemcu_sensors_pump/nodemcu_sensors_pump.ino).

The NodeMCU reads soil moisture (capacitive sensor, raw ADC -> %), air temperature and humidity (DHT11)
and switches a relay-driven pump. Logic, with hysteresis so the pump does not chatter:

    pump turns ON  when moisture < ON_THRESHOLD (default 35 %)
    pump turns OFF when moisture > OFF_THRESHOLD (default 60 %) or after MAX_ON_SECONDS
    ON threshold is raised by 5 % when temperature > 32 C and humidity < 40 % (high evaporation)
    leaf-wetness disease risk = "high" when humidity > 85 % and 15 C <= temp <= 28 C
"""
from __future__ import annotations

from dataclasses import dataclass, field


def adc_to_moisture(raw: int, dry: int = 3200, wet: int = 1300) -> float:
    """Capacitive sensor: higher ADC = drier. Linear calibration clipped to 0-100 %."""
    pct = (dry - raw) / (dry - wet) * 100.0
    return max(0.0, min(100.0, pct))


@dataclass
class IrrigationController:
    on_threshold: float = 35.0
    off_threshold: float = 60.0
    max_on_seconds: int = 600
    pump_on: bool = False
    on_since: float | None = None
    log: list = field(default_factory=list)

    def effective_on_threshold(self, temp_c: float, humidity: float) -> float:
        return self.on_threshold + (5.0 if temp_c > 32 and humidity < 40 else 0.0)

    def step(self, t: float, moisture: float, temp_c: float, humidity: float) -> bool:
        on_th = self.effective_on_threshold(temp_c, humidity)
        if not self.pump_on and moisture < on_th:
            self.pump_on, self.on_since = True, t
        elif self.pump_on and (moisture > self.off_threshold or t - (self.on_since or t) >= self.max_on_seconds):
            self.pump_on, self.on_since = False, None
        self.log.append((t, moisture, temp_c, humidity, self.pump_on))
        return self.pump_on


def disease_risk(temp_c: float, humidity: float) -> str:
    if humidity > 85 and 15 <= temp_c <= 28:
        return "high"
    if humidity > 70:
        return "medium"
    return "low"
