"""Audit game (costly state verification). All closed forms used in the report live here, with tests in tests/test_game.py.

Setting, per CANDIDATE (an applicant who holds a concealable risk factor), money in INR over the term:
  dpi    premium saved by concealing (price of the true record minus price of the declared record)
  r      trigger recall: P(flagged | concealer)         f   P(flagged | honest)       v  exam accuracy: P(detect | verified concealer)
  sigma  probability the insurer runs the flag-and-verify policy      c  cost per verified applicant
  L      TOTAL loss on detection = forfeited saving + surcharge (so surcharge F = L - dpi)
Applicant payoff of concealing minus disclosing = (1 - sigma r v) dpi - sigma r v F = dpi - sigma r v L.
Insurer payoff of running the policy minus not, when a share theta of candidates conceal:
        theta r v dpi - c [ theta r + (1 - theta) f ].
"""


def detection_prob(r, v):
    return r * v


def deterrence_threshold(dpi, r, v):
    """P* = dpi / (r v): smallest TOTAL loss on detection making truthful disclosure a best response even when the insurer
    always runs the policy (sigma = 1)."""
    return dpi / (r * v)


def deterrence_multiple(r, v):
    """P* / dpi = 1 / (r v): the penalty must be this many times the saving. Scale-free."""
    return 1.0 / (r * v)


def required_surcharge(dpi, r, v):
    """F* = P* - dpi: surcharge needed on top of forfeiting the saving."""
    return deterrence_threshold(dpi, r, v) - dpi


def applicant_gain_from_concealing(dpi, L, sigma, r, v):
    return dpi - sigma * r * v * L


def sigma_star(dpi, L, r, v):
    """Verification probability that makes the applicant indifferent. Valid (<= 1) iff L >= P*."""
    return dpi / (r * v * L)


def insurer_gain_from_policy(theta, dpi, r, f, v, c):
    return theta * r * v * dpi - c * (theta * r + (1 - theta) * f)


def theta_star(dpi, r, f, v, c):
    """Concealment probability that makes the insurer indifferent. Exists only if the policy can ever pay: v*dpi > c."""
    if v * dpi <= c:
        return None
    return c * f / (r * (v * dpi - c) + c * f)


def verify_flagged_if(c, precision, v, dpi):
    """Decision rule for a flagged applicant: expected recovered premium exceeds the cost."""
    return c < precision * v * dpi
