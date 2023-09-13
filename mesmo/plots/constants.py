import enum


class ValueUnitLabels(str, enum.Enum):
    # TODO: Convert into base units
    WATT = "MW"
    VOLT_AMPERE_REACTIVE = "MVAr"
    VOLT_AMPERE = "MVA"


class ValueLabels(str, enum.Enum):
    TIME = "Date/Time"
    DERS = "DERs"
    ACTIVE_POWER = "Active power"
    REACTIVE_POWER = "Reactive power"
    APPARENT_POWER = "Apparent power"
