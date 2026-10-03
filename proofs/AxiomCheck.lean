import Proofs

/-! Run by tools/check_proofs.sh: every headline theorem must depend only on
propext / Classical.choice / Quot.sound. -/

open Immerdink

-- L2 (CRT.lean)
#print axioms CRT.x_lt_P
#print axioms CRT.xp_lt_two_P
#print axioms CRT.xp_lt_word
#print axioms CRT.x_N_modEq
#print axioms CRT.exists_q
#print axioms CRT.z_eq
#print axioms CRT.z_eq_of_even
#print axioms CRT.z_eq_floor
