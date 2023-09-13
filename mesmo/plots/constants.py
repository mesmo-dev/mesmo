import enum


class ValueUnitLabels(str, enum.Enum):
    # TODO: Convert into base units
    WATT = "MW"
    VOLT_AMPERE_REACTIVE = "MVAr"
    VOLT_AMPERE = "MVA"
    VOLT_PER_UNIT = "V (per-unit)"


class ValueLabels(str, enum.Enum):
    TIME = "Date/Time"
    DERS = "DERs"
    NODES = "Nodes"
    ACTIVE_POWER = "Active power"
    REACTIVE_POWER = "Reactive power"
    APPARENT_POWER = "Apparent power"
    VOLTAGE = "Voltage"
