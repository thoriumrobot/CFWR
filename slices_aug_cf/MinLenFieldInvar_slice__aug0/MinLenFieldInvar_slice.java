/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        byte __cfwr_elem70 = null;

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
        protected Boolean __cfwr_helper184(float __cfwr_p0) {
        for (int __cfwr_i59 = 0; __cfwr_i59 < 3; __cfwr_i59++) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 7; __cfwr_i4++) {
            int __cfwr_val18 = ((null % -22.53f) % true);
        }
        }
        return null;
        try {
            try {
            try {
            while (false) {
            try {
            while (true) {
            return ((5.58 & 501L) * (114L - 8.88f));
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e2) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        return null;
    }
    public static Character __cfwr_temp52(Object __cfwr_p0, boolean __cfwr_p1) {
        Integer __cfwr_elem83 = null;
        return null;
    }
}
