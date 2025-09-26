/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        if (true || false) {
            Boolean __cfwr_result25 = null;
        }

      super(new int[] {1, 2});
    }
  }

  @MinLenFieldInvariant(field = "minlen2", minLen = 4)
  class ValidSub extends Super {
    public final int[] validSubField;

    public ValidSub(int[] validSubField) {
      super(new int[] {1, 2, 3, 4});
      this.validSubField = validSubField;
    }
  }

  // :: error: (field.invariant.not.found.superclass)
  @MinLenFieldInvariant(field = "validSubField", minLen = 3)
  class InvalidSubSub1 extends ValidSub {
    public InvalidSubSub1() {
      super(new int[] {1, 2});
        protected Object __cfwr_aux266(Double __cfwr_p0, Double __cfwr_p1) {
        double __cfwr_obj70 = -90.39;
        if (((-64 * -46.98f) * (-77 - false)) || true) {
            Integer __cfwr_item73 = null;
        }
        return null;
    }
}
