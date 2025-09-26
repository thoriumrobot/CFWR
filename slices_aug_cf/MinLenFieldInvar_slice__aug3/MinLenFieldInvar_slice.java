/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        try {
            while (false) {
            Double __cfwr_val80 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }


        while (((true / null) & true)) {
            while (false) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 9; __cfwr_i92++) {
            Float __cfwr_temp90 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
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
        Character __cfwr_helper95(Integer __cfwr_p0, Long __cfwr_p1) {
        return 38.14;
        Float __cfwr_elem76 = null;
        return null;
    }
    static byte __cfwr_temp662(long __cfwr_p0) {
        return null;
        return null;
    }
}
