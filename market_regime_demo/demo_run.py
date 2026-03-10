from my_project.library.str_mar import (
    market_features,
    market_state_vector,
    market_report,
    regime_interpretation,
    ml_next_regime_model
)

state = market_state_vector("^NSEI", "2010-01-01")
mrkt_rp = market_report("^NSEI", "2010-01-01")
mrkt_rp_p = market_features("^NSEI", "2010-01-01")

print("\nMarket State Vector:")
for k, v in state.items():
    print(k, ":", v)

print("\nRegime Interpretation:")
print(regime_interpretation("^NSEI", "2010-01-01"))

print("\nRegime Survival:")
print(regime_survival_analysis("^NSEI", "2010-01-01"))

print("\nML Next Regime Performance:")
print(ml_next_regime_model("^NSEI", "2010-01-01"))

