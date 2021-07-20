from openfisca_uk.api import *

UBI_AMOUNT = 20 * 52

# Costs £68.8bn in year 1

class UBI(Variable):
    value_type = float
    entity = Person
    definition_period = YEAR

    def formula(person, period, parameters):
        return 20 * 52

class tax(BASELINE_VARIABLES.tax):
    def formula(person, period, parameters):
        return BASELINE_VARIABLES.tax.formula(person, period, parameters) - person("UBI", period)


ubi_reform = reforms.structural.new_variable(UBI), reforms.structural.restructure(tax)