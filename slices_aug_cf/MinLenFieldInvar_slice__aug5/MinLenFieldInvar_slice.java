/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        Float __cfwr_node67 = null;

      super(new int[] {1, 2});
    }
  }

  @MinLenFieldInvariant(field = "minlen2", minLen = 4)
  class ValidSub extends Super {
    public final int[] validSubField;

    p
        char __cfwr_temp2 = '6';
ublic ValidSub(int[] validSubField) {
      super(new int[] {1, 2, 3, 4});
      this.validSubField = validSubField;
    }
  }

  // :: error: (field.invariant.not.found.superclass)
  @MinLenFieldInvariant(field = "validSubField", minLen = 3)
  class InvalidSubSub1 extends ValidSub {
    public InvalidSubSub1() {
      super(new int[] {1, 2});
        protected Boolean __cfwr_calc156(Character __cfwr_p0) {
        try {
            return "hello58";
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        return null;
    }
    protected static Boolean __cfwr_handle152(double __cfwr_p0) {
        if (true || ((-915L % 'h') + (-768 % -44.35f))) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 4; __cfwr_i15++) {
            return null;
        }
        }
        return null;
        Float __cfwr_item30 = null;
        return null;
    }
}
