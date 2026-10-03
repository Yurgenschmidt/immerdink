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
