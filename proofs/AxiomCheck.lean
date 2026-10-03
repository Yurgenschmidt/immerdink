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

-- L4 (Chunk.lean)
#print axioms Chunk.satFold_eq
#print axioms Chunk.AccTree.eval_eq_sum
#print axioms Chunk.partial_sum_bound
#print axioms Chunk.satFold_chunk_eq_sum
#print axioms Chunk.tree_chunk_eq_sum
#print axioms Chunk.epi_modEq_total
#print axioms Chunk.epi_mem_range
#print axioms Chunk.carry_correct
#print axioms Chunk.epi_input_range
#print axioms Chunk.design_constants
#print axioms Chunk.epi_input_int32

-- L1 (Scaling.lean)
#print axioms Scaling.l2_add_le
#print axioms Scaling.l1_le_sqrt_mul_l2
#print axioms Scaling.two_pow_le_S
#print axioms Scaling.exists_max_rule
#print axioms Scaling.not_rule_succ
#print axioms Scaling.exists_u_rn
#print axioms Scaling.exists_u_trunc
#print axioms Scaling.no_rule_of_R_zero
#print axioms Scaling.budget_rn
#print axioms Scaling.budget_trunc
#print axioms Scaling.budget_rn_int
#print axioms Scaling.budget_trunc_int
#print axioms Scaling.crt_precondition
#print axioms Scaling.sigma_bound_rn
#print axioms Scaling.sigma_bound_trunc
#print axioms Scaling.row_rn
#print axioms Scaling.row_trunc
#print axioms Scaling.abs_rne_sub
#print axioms Scaling.abs_truncZ_le
#print axioms Scaling.abs_truncZ_sub_lt

-- L3 (Error.lean)
#print axioms Error.l1_form
#print axioms Error.normwise_core
#print axioms Error.normwise_rn
#print axioms Error.normwise_trunc
#print axioms Error.end_to_end_rn
#print axioms Error.end_to_end_trunc
#print axioms Error.zero_row
#print axioms Error.q_zero_rn
