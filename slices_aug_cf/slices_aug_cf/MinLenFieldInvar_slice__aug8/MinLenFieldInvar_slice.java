/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        return null;

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
        private static String __cfwr_aux789() {
        while (false) {
            Boolean __cfwr_node89 = null;
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        return "hello92";
    }
}
