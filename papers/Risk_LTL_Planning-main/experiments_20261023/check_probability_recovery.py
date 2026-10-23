"""Regression for amplification of tiny negative LP occupations."""
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'reproduction/carla_local_risk'))
sys.path.insert(0,str(REPO/'reproduction/adaptive_risk_ltl'))
from dynamic_certificate import policy_from_occupation
occupation=np.array([[1.,0.],[-1e-9,1.01e-9],[0.,0.]])
policy=policy_from_occupation(occupation)
assert np.min(policy)>=0 and np.max(policy)<=1
assert np.allclose(policy.sum(axis=1),1)
assert np.allclose(policy[1],[0,1])
assert np.allclose(policy[2],[1,0])
print('PASS: tiny negative occupation and zero-mass state recover valid probabilities')
