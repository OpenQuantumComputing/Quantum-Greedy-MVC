import Mathlib

open Set

namespace ProjectedMixerLightCone

variable {V : Type*}

def nbr (Adj : V → V → Prop) (i : V) : Set V :=
  {j | Adj i j ∨ Adj j i}

def ball (Adj : V → V → Prop) (v : V) : ℕ → Set V
  | 0 => {v}
  | n + 1 =>
      ball Adj v n ∪
        {x | ∃ y ∈ ball Adj v n, x ∈ nbr Adj y}

def innerBall (Adj : V → V → Prop) (v : V) : ℕ → Set V
  | 0 => ∅
  | n + 1 => ball Adj v n

structure PTerm (V : Type*) where
  support : Set V
  xySupport : Set V

def initial (v : V) : PTerm V where
  support := {v}
  xySupport := ∅

def Step (Adj : V → V → Prop) (q r : PTerm V) : Prop :=
  ∃ i T,
    T ⊆ nbr Adj i ∧
    (i ∈ q.support ∨ ∃ j ∈ T, j ∈ q.xySupport) ∧
    r.support ⊆ q.support ∪ ({i} ∪ T) ∧
    r.xySupport ⊆ q.xySupport ∪ {i}

inductive Reach (Adj : V → V → Prop) (v : V) :
    ℕ → PTerm V → Prop
  | zero :
      Reach Adj v 0 (initial v)
  | succ {n q r} :
      Reach Adj v n q →
      Step Adj q r →
      Reach Adj v (n + 1) r

lemma mem_nbr_symm
    (Adj : V → V → Prop) {i j : V}
    (h : j ∈ nbr Adj i) :
    i ∈ nbr Adj j := by
  change Adj i j ∨ Adj j i at h
  change Adj j i ∨ Adj i j
  exact h.elim Or.inr Or.inl

lemma ball_subset_succ
    (Adj : V → V → Prop) (v : V) (n : ℕ) :
    ball Adj v n ⊆ ball Adj v (n + 1) := by
  intro x hx
  exact Or.inl hx

lemma innerBall_subset_ball
    (Adj : V → V → Prop) (v : V) (n : ℕ) :
    innerBall Adj v n ⊆ ball Adj v n := by
  cases n with
  | zero =>
      simp [innerBall]
  | succ n =>
      simpa [innerBall] using ball_subset_succ Adj v n

theorem reach_lightCone
    (Adj : V → V → Prop) (v : V)
    {n : ℕ} {q : PTerm V}
    (h : Reach Adj v n q) :
    q.support ⊆ ball Adj v n ∧
    q.xySupport ⊆ innerBall Adj v n := by
  induction h with
  | zero =>
      simp [initial, ball, innerBall]
  | @succ n q r hq hstep ih =>
      rcases hstep with
        ⟨i, T, hT, hactive, hsupp, hxy⟩

      have hi : i ∈ ball Adj v n := by
        rcases hactive with hiq | ⟨j, hjT, hjxy⟩
        · exact ih.1 hiq
        · cases n with
          | zero =>
              have hjEmpty : j ∈ (∅ : Set V) := by
                simpa [innerBall] using ih.2 hjxy
              exact hjEmpty.elim
          | succ n =>
              have hjBall : j ∈ ball Adj v n := by
                simpa [innerBall] using ih.2 hjxy
              have hji : j ∈ nbr Adj i := hT hjT
              have hij : i ∈ nbr Adj j :=
                mem_nbr_symm Adj hji
              exact Or.inr ⟨j, hjBall, hij⟩

      constructor
      · intro x hxr
        rcases hsupp hxr with hxq | hxit
        · exact ball_subset_succ Adj v n (ih.1 hxq)
        · rcases hxit with hxi | hxT
          · have hxi' : x = i := by
              simpa using hxi
            subst x
            exact ball_subset_succ Adj v n hi
          · exact Or.inr ⟨i, hi, hT hxT⟩
      · intro x hxr
        change x ∈ ball Adj v n
        rcases hxy hxr with hxq | hxi
        · exact innerBall_subset_ball Adj v n (ih.2 hxq)
        · have hxi' : x = i := by
            simpa using hxi
          subst x
          exact hi

theorem operator_support_lightCone
    (Adj : V → V → Prop) (v : V)
    {n : ℕ} {q : PTerm V}
    (h : Reach Adj v n q) :
    q.support ⊆ ball Adj v n :=
  (reach_lightCone Adj v h).1

structure FinPauli (V : Type*) where
  xSupport : Finset V
  zSupport : Finset V

namespace FinPauli

variable {V : Type*} [DecidableEq V]

def toTerm (q : FinPauli V) : PTerm V where
  support := ↑(q.xSupport ∪ q.zSupport)
  xySupport := ↑q.xSupport

def finXor (a b : Finset V) : Finset V :=
  (a \ b) ∪ (b \ a)

def localProduct
    (i : V) (T : Finset V) (q : FinPauli V) :
    FinPauli V where
  xSupport := finXor {i} q.xSupport
  zSupport := finXor T q.zSupport

def localAntiCount
    (i : V) (T : Finset V) (q : FinPauli V) : ℕ :=
  (if i ∈ q.zSupport then 1 else 0) +
    (T ∩ q.xSupport).card

def LocalAnticommutes
    (i : V) (T : Finset V) (q : FinPauli V) : Prop :=
  Odd (localAntiCount i T q)

lemma localAnticommutes_activation
    {i : V} {T : Finset V} {q : FinPauli V}
    (hanti : LocalAnticommutes i T q) :
    i ∈ q.zSupport ∨
      ∃ j ∈ T, j ∈ q.xSupport := by
  by_cases hi : i ∈ q.zSupport
  · exact Or.inl hi
  · by_cases hj : ∃ j ∈ T, j ∈ q.xSupport
    · exact Or.inr hj
    · have hInter : T ∩ q.xSupport = ∅ := by
        ext j
        constructor
        · intro hjInter
          have hj' := Finset.mem_inter.mp hjInter
          exact (hj ⟨j, hj'.1, hj'.2⟩).elim
        · intro hjEmpty
          simp at hjEmpty
      have hzero : localAntiCount i T q = 0 := by
        simp [localAntiCount, hi, hInter]
      change Odd (localAntiCount i T q) at hanti
      rw [hzero] at hanti
      rcases hanti with ⟨k, hk⟩
      omega

theorem localCommutator_step
    (Adj : V → V → Prop)
    (i : V)
    (T : Finset V)
    (q : FinPauli V)
    (hT : ∀ j ∈ T, j ∈ nbr Adj i)
    (hanti : LocalAnticommutes i T q) :
    Step Adj q.toTerm (localProduct i T q).toTerm := by
  refine ⟨i, (↑T : Set V), ?_, ?_, ?_, ?_⟩
  · intro j hj
    exact hT j hj
  · rcases localAnticommutes_activation hanti with
      hi | ⟨j, hjT, hjX⟩
    · left
      change i ∈ (↑(q.xSupport ∪ q.zSupport) : Set V)
      exact Finset.mem_union.mpr (Or.inr hi)
    · right
      refine ⟨j, hjT, ?_⟩
      change j ∈ (↑q.xSupport : Set V)
      exact hjX
  · intro x hx
    simp [toTerm, localProduct, finXor] at hx ⊢
    aesop
  · intro x hx
    simp [toTerm, localProduct, finXor] at hx ⊢
    aesop

end FinPauli

end ProjectedMixerLightCone
