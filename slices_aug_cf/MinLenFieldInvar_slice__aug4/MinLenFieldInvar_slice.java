/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        for (int __cfwr_i98 = 0; __cfwr_i98 < 9; __cfwr_i98++) {
            return null;
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
        Double __cfwr_temp735(byte __cfwr_p0, short __cfwr_p1) {
        while (true) {
            long __cfwr_elem55 = 934L;
            break; // Prevent infinite loops
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        while (false) {
            while (((-74 % null) * 'r')) {
            if (true && true) {
            try {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            short __cfwr_node72 = null;
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
